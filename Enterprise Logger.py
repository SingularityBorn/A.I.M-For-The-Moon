"""
Enterprise Logger System

A comprehensive, high-performance logging system with advanced features including
log rotation, batching, performance metrics, structured logging, and correlation tracking.

Features:
- Multiple output formats (JSON, structured, text)
- Asynchronous file operations with batching
- Automatic log rotation with size limits
- Performance metrics and monitoring
- Correlation ID tracking
- Error serialization with cause chains
- Graceful shutdown handling
- Memory usage monitoring
- Worker thread support for CPU-intensive operations
"""

import asyncio
import json
import os
import signal
import sys
import threading
import traceback
import uuid
import zlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum, IntEnum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Protocol, runtime_checkable
from concurrent.futures import ThreadPoolExecutor
import psutil
import aiofiles
from aiofiles import os as aio_os


class LogLevel(IntEnum):
    """Log severity levels."""
    DEBUG = 0
    INFO = 1
    WARN = 2
    ERROR = 3
    FATAL = 4


class TimestampFormat(Enum):
    """Timestamp formatting options."""
    ISO = "iso"
    EPOCH = "epoch"
    LOCAL = "local"


class LogFormat(Enum):
    """Log output formats."""
    JSON = "json"
    TEXT = "text" 
    STRUCTURED = "structured"


@dataclass(frozen=True)
class SerializedError:
    """Serialized error information."""
    name: str
    message: str
    stack: Optional[str] = None
    code: Optional[Union[str, int]] = None
    cause: Optional['SerializedError'] = None


@dataclass(frozen=True)
class PerformanceMetrics:
    """Performance measurement data."""
    duration: Optional[float] = None
    memory_usage: Optional[Dict[str, int]] = None
    cpu_percent: Optional[float] = None


@dataclass(frozen=True)
class LogEntry:
    """Individual log entry."""
    timestamp: datetime
    level: LogLevel
    component: str
    message: str
    correlation_id: Optional[str] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    error: Optional[SerializedError] = None
    performance: Optional[PerformanceMetrics] = None


@dataclass
class LoggerConfig:
    """Logger configuration options."""
    min_level: LogLevel = LogLevel.INFO
    enable_console: bool = True
    log_file: Optional[Path] = None
    format: LogFormat = LogFormat.JSON
    additional_fields: Optional[Dict[str, Any]] = None
    rotate_size: int = 50 * 1024 * 1024  # 50MB
    max_files: int = 10
    flush_interval: float = 5.0  # seconds
    enable_batching: bool = True
    batch_size: int = 100
    enable_compression: bool = False
    enable_worker: bool = False
    max_memory_buffer: int = 100 * 1024 * 1024  # 100MB
    enable_metrics: bool = True
    timestamp_format: TimestampFormat = TimestampFormat.ISO
    enable_correlation: bool = False

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.additional_fields is None:
            self.additional_fields = {}
        
        if self.rotate_size <= 0:
            raise ValueError("rotate_size must be greater than 0")
        if self.max_files <= 0:
            raise ValueError("max_files must be greater than 0")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")
        if self.flush_interval <= 0:
            raise ValueError("flush_interval must be greater than 0")


@dataclass
class LoggerMetrics:
    """Logger performance metrics."""
    total_logs: int = 0
    logs_by_level: Dict[LogLevel, int] = field(default_factory=lambda: {
        LogLevel.DEBUG: 0,
        LogLevel.INFO: 0,
        LogLevel.WARN: 0,
        LogLevel.ERROR: 0,
        LogLevel.FATAL: 0
    })
    avg_processing_time: float = 0.0
    errors_count: int = 0
    memory_usage: int = 0
    uptime: float = 0.0


class AsyncFileWriter:
    """Asynchronous file writer with rotation support."""
    
    def __init__(self, file_path: Path, max_size: int, max_files: int):
        self.file_path = file_path
        self.max_size = max_size
        self.max_files = max_files
        self.current_size = 0
        self.lock = asyncio.Lock()
        self._file_handle: Optional[aiofiles.threadpool.text.AsyncTextIOWrapper] = None
        
    async def initialize(self):
        """Initialize the file writer."""
        # Ensure directory exists
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get current file size
        if self.file_path.exists():
            self.current_size = self.file_path.stat().st_size
        else:
            self.current_size = 0
            
        # Open file for appending
        self._file_handle = await aiofiles.open(self.file_path, 'a', encoding='utf-8')
    
    async def write(self, content: str) -> None:
        """Write content to file with rotation check."""
        async with self.lock:
            if not self._file_handle:
                await self.initialize()
                
            content_size = len(content.encode('utf-8'))
            
            # Check if rotation is needed
            if self.current_size + content_size > self.max_size:
                await self._rotate()
                self.current_size = 0
                
            await self._file_handle.write(content)
            await self._file_handle.flush()
            self.current_size += content_size
    
    async def _rotate(self):
        """Rotate log files."""
        if self._file_handle:
            await self._file_handle.close()
            
        # Rotate existing files
        for i in range(self.max_files - 1, 0, -1):
            old_file = Path(f"{self.file_path}.{i}")
            new_file = Path(f"{self.file_path}.{i + 1}")
            
            if old_file.exists():
                if i == self.max_files - 1:
                    old_file.unlink()  # Delete oldest file
                else:
                    old_file.rename(new_file)
        
        # Move current file to .1
        if self.file_path.exists():
            self.file_path.rename(Path(f"{self.file_path}.1"))
            
        # Reopen new file
        self._file_handle = await aiofiles.open(self.file_path, 'a', encoding='utf-8')
    
    async def close(self):
        """Close the file writer."""
        async with self.lock:
            if self._file_handle:
                await self._file_handle.close()
                self._file_handle = None


class Logger:
    """
    Enterprise Logger System
    
    High-performance, feature-rich logging system with enterprise capabilities
    including structured logging, performance monitoring, and advanced file management.
    """
    
    def __init__(self, component: str, config: Optional[LoggerConfig] = None):
        if not component or not component.strip():
            raise ValueError("Component name is required and cannot be empty")
            
        self.component = component.strip()
        self.config = config or LoggerConfig()
        self.start_time = datetime.now(timezone.utc)
        self.metrics = LoggerMetrics()
        self.processing_times: List[float] = []
        
        # Internal state
        self.log_buffer: List[LogEntry] = []
        self.is_closing = False
        self.file_writer: Optional[AsyncFileWriter] = None
        self.flush_task: Optional[asyncio.Task] = None
        self.executor = ThreadPoolExecutor(max_workers=2) if config and config.enable_worker else None
        
        # Locks for thread safety
        self.buffer_lock = asyncio.Lock()
        self.metrics_lock = threading.Lock()
        
        # Initialize async components
        asyncio.create_task(self._initialize_async())
        
        # Setup graceful shutdown
        self._setup_graceful_shutdown()
    
    async def _initialize_async(self):
        """Initialize asynchronous components."""
        try:
            if self.config.log_file:
                self.file_writer = AsyncFileWriter(
                    self.config.log_file,
                    self.config.rotate_size,
                    self.config.max_files
                )
                await self.file_writer.initialize()
            
            if self.config.enable_batching:
                self._start_batch_flushing()
                
        except Exception as e:
            print(f"Failed to initialize logger for component {self.component}: {e}")
    
    def _setup_graceful_shutdown(self):
        """Setup graceful shutdown handlers."""
        def signal_handler(signum, frame):
            asyncio.create_task(self.close())
            
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def _start_batch_flushing(self):
        """Start the batch flushing task."""
        async def flush_periodically():
            while not self.is_closing:
                try:
                    await asyncio.sleep(self.config.flush_interval)
                    if self.log_buffer:
                        await self._flush_buffer()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    print(f"Error in batch flushing: {e}")
        
        self.flush_task = asyncio.create_task(flush_periodically())
    
    def debug(self, message: str, data: Optional[Dict[str, Any]] = None, 
              correlation_id: Optional[str] = None):
        """Log debug message."""
        asyncio.create_task(self._log(LogLevel.DEBUG, message, data, None, correlation_id))
    
    def info(self, message: str, data: Optional[Dict[str, Any]] = None,
             correlation_id: Optional[str] = None):
        """Log info message."""
        asyncio.create_task(self._log(LogLevel.INFO, message, data, None, correlation_id))
    
    def warn(self, message: str, data: Optional[Dict[str, Any]] = None,
             correlation_id: Optional[str] = None):
        """Log warning message."""
        asyncio.create_task(self._log(LogLevel.WARN, message, data, None, correlation_id))
    
    def error(self, message: str, error: Optional[Exception] = None,
              data: Optional[Dict[str, Any]] = None, correlation_id: Optional[str] = None):
        """Log error message."""
        asyncio.create_task(self._log(LogLevel.ERROR, message, data, error, correlation_id))
    
    def fatal(self, message: str, error: Optional[Exception] = None,
              data: Optional[Dict[str, Any]] = None, correlation_id: Optional[str] = None):
        """Log fatal message with immediate flush."""
        task = asyncio.create_task(self._log(LogLevel.FATAL, message, data, error, correlation_id))
        # For fatal errors, try to flush immediately
        asyncio.create_task(self._flush_buffer())
    
    def performance(self, message: str, duration: float, 
                   data: Optional[Dict[str, Any]] = None, correlation_id: Optional[str] = None):
        """Log performance metrics."""
        try:
            # Get current process info
            process = psutil.Process()
            memory_info = process.memory_info()
            
            performance_metrics = PerformanceMetrics(
                duration=duration,
                memory_usage={
                    'rss': memory_info.rss,
                    'vms': memory_info.vms
                },
                cpu_percent=process.cpu_percent()
            )
            
            asyncio.create_task(self._log(
                LogLevel.INFO, message, data, None, correlation_id, performance_metrics
            ))
        except Exception as e:
            # Fallback to basic logging if psutil fails
            asyncio.create_task(self._log(LogLevel.INFO, message, data))
    
    async def _log(self, level: LogLevel, message: str, data: Optional[Dict[str, Any]] = None,
                   error: Optional[Exception] = None, correlation_id: Optional[str] = None,
                   performance: Optional[PerformanceMetrics] = None):
        """Internal logging method."""
        if self.is_closing or not self._should_log(level):
            return
            
        start_time = datetime.now(timezone.utc)
        
        try:
            entry = self._create_log_entry(level, message, data, error, correlation_id, performance)
            
            if self.config.enable_batching and level < LogLevel.FATAL:
                await self._add_to_buffer(entry)
            else:
                await self._write_log_immediate(entry)
                
            self._update_metrics(level, start_time)
            
        except Exception as log_error:
            print(f"Error creating log entry for component {self.component}: {log_error}")
            self._update_metrics(LogLevel.ERROR, start_time)
    
    def _create_log_entry(self, level: LogLevel, message: str, data: Optional[Dict[str, Any]] = None,
                         error: Optional[Exception] = None, correlation_id: Optional[str] = None,
                         performance: Optional[PerformanceMetrics] = None) -> LogEntry:
        """Create a log entry."""
        # Merge additional fields with provided data
        merged_data = {**(self.config.additional_fields or {})}
        if data:
            merged_data.update(data)
        
        return LogEntry(
            timestamp=datetime.now(timezone.utc),
            level=level,
            component=self.component,
            message=message.strip(),
            correlation_id=correlation_id or (self._generate_correlation_id() if self.config.enable_correlation else None),
            data=merged_data if merged_data else None,
            error=self._serialize_error(error) if error else None,
            performance=performance
        )
    
    def _generate_correlation_id(self) -> str:
        """Generate a unique correlation ID."""
        return f"{int(datetime.now().timestamp() * 1000)}-{uuid.uuid4().hex[:8]}"
    
    def _serialize_error(self, error: Exception) -> SerializedError:
        """Serialize an exception to a structured format."""
        serialized = SerializedError(
            name=error.__class__.__name__,
            message=str(error),
            stack=traceback.format_exc()
        )
        
        # Handle error codes
        if hasattr(error, 'code'):
            serialized = SerializedError(
                name=serialized.name,
                message=serialized.message,
                stack=serialized.stack,
                code=error.code
            )
        
        # Handle nested errors (cause chains)
        if hasattr(error, '__cause__') and error.__cause__:
            serialized = SerializedError(
                name=serialized.name,
                message=serialized.message,
                stack=serialized.stack,
                code=serialized.code,
                cause=self._serialize_error(error.__cause__)
            )
        
        return serialized
    
    def _should_log(self, level: LogLevel) -> bool:
        """Check if the log level should be processed."""
        return level >= self.config.min_level
    
    async def _add_to_buffer(self, entry: LogEntry):
        """Add entry to buffer with size and memory checks."""
        async with self.buffer_lock:
            self.log_buffer.append(entry)
            
            # Check if we should flush based on buffer size
            if len(self.log_buffer) >= self.config.batch_size:
                await self._flush_buffer()
                return
            
            # Check memory usage
            buffer_memory = self._estimate_buffer_memory()
            if buffer_memory > self.config.max_memory_buffer:
                await self._flush_buffer()
    
    def _estimate_buffer_memory(self) -> int:
        """Estimate memory usage of the log buffer."""
        return len(self.log_buffer) * 1024  # Rough estimation
    
    async def _flush_buffer(self):
        """Flush all entries in the buffer."""
        async with self.buffer_lock:
            if not self.log_buffer:
                return
                
            entries_to_flush = self.log_buffer.copy()
            self.log_buffer.clear()
        
        # Write all entries
        tasks = [self._write_log_immediate(entry) for entry in entries_to_flush]
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _write_log_immediate(self, entry: LogEntry):
        """Write a log entry immediately to all configured outputs."""
        formatted_entry = self._format_entry(entry)
        
        tasks = []
        
        if self.config.enable_console:
            tasks.append(self._write_to_console(entry, formatted_entry))
        
        if self.config.log_file and self.file_writer:
            tasks.append(self._write_to_file(formatted_entry))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    def _format_entry(self, entry: LogEntry) -> str:
        """Format a log entry according to the configured format."""
        timestamp = self._format_timestamp(entry.timestamp)
        
        if self.config.format == LogFormat.JSON:
            # Convert to dictionary and serialize
            entry_dict = asdict(entry)
            entry_dict['timestamp'] = timestamp
            entry_dict['level'] = entry.level.name
            return json.dumps(entry_dict, default=str)
        
        elif self.config.format == LogFormat.STRUCTURED:
            structured = {
                '@timestamp': timestamp,
                '@level': entry.level.name,
                '@component': entry.component,
                '@message': entry.message,
                '@correlation_id': entry.correlation_id,
            }
            
            if entry.data:
                structured.update(entry.data)
            if entry.error:
                structured['@error'] = asdict(entry.error)
            if entry.performance:
                structured['@performance'] = asdict(entry.performance)
                
            return json.dumps(structured, default=str)
        
        else:  # TEXT format
            message = f"[{timestamp}] {entry.level.name.ljust(5)} [{entry.component}] {entry.message}"
            
            if entry.correlation_id:
                message += f" [{entry.correlation_id}]"
            
            if entry.data:
                message += f"\n  Data: {json.dumps(entry.data, indent=2, default=str)}"
            
            if entry.error:
                message += f"\n  Error: {entry.error.name}: {entry.error.message}"
                if entry.error.stack:
                    message += f"\n  Stack: {entry.error.stack}"
            
            if entry.performance:
                message += f"\n  Performance: {json.dumps(asdict(entry.performance), indent=2, default=str)}"
            
            return message
    
    def _format_timestamp(self, timestamp: datetime) -> str:
        """Format timestamp according to configuration."""
        if self.config.timestamp_format == TimestampFormat.EPOCH:
            return str(int(timestamp.timestamp()))
        elif self.config.timestamp_format == TimestampFormat.LOCAL:
            return timestamp.replace(tzinfo=timezone.utc).astimezone().strftime('%Y-%m-%d %H:%M:%S')
        else:  # ISO format
            return timestamp.isoformat()
    
    async def _write_to_console(self, entry: LogEntry, formatted_entry: str):
        """Write to console output."""
        if entry.level >= LogLevel.ERROR:
            print(formatted_entry, file=sys.stderr)
        else:
            print(formatted_entry)
    
    async def _write_to_file(self, formatted_entry: str):
        """Write to file output."""
        if self.file_writer:
            await self.file_writer.write(formatted_entry + '\n')
    
    def _update_metrics(self, level: LogLevel, start_time: datetime):
        """Update logger metrics."""
        if not self.config.enable_metrics:
            return
            
        with self.metrics_lock:
            self.metrics.total_logs += 1
            self.metrics.logs_by_level[level] += 1
            
            if level >= LogLevel.ERROR:
                self.metrics.errors_count += 1
            
            # Calculate processing time
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            self.processing_times.append(processing_time)
            
            # Keep only last 1000 measurements
            if len(self.processing_times) > 1000:
                self.processing_times = self.processing_times[-1000:]
            
            self.metrics.avg_processing_time = sum(self.processing_times) / len(self.processing_times)
            
            # Update memory usage
            try:
                process = psutil.Process()
                self.metrics.memory_usage = process.memory_info().rss
            except:
                pass
            
            # Update uptime
            self.metrics.uptime = (datetime.now(timezone.utc) - self.start_time).total_seconds()
    
    def get_metrics(self) -> LoggerMetrics:
        """Get current logger metrics."""
        with self.metrics_lock:
            return LoggerMetrics(
                total_logs=self.metrics.total_logs,
                logs_by_level=self.metrics.logs_by_level.copy(),
                avg_processing_time=self.metrics.avg_processing_time,
                errors_count=self.metrics.errors_count,
                memory_usage=self.metrics.memory_usage,
                uptime=self.metrics.uptime
            )
    
    async def close(self):
        """Close the logger and clean up resources."""
        if self.is_closing:
            return
            
        self.is_closing = True
        
        try:
            # Cancel flush task
            if self.flush_task:
                self.flush_task.cancel()
                try:
                    await self.flush_task
                except asyncio.CancelledError:
                    pass
            
            # Flush remaining buffer
            await self._flush_buffer()
            
            # Close file writer
            if self.file_writer:
                await self.file_writer.close()
            
            # Shutdown executor
            if self.executor:
                self.executor.shutdown(wait=True)
                
        except Exception as e:
            print(f"Error closing logger for component {self.component}: {e}")
            raise


# Factory function for easy logger creation
def create_logger(component: str, **config_kwargs) -> Logger:
    """Create a logger with optional configuration parameters."""
    config = LoggerConfig(**config_kwargs)
    return Logger(component, config)


# Export main classes and functions
__all__ = [
    'Logger',
    'LoggerConfig', 
    'LoggerMetrics',
    'LogLevel',
    'LogFormat',
    'TimestampFormat',
    'LogEntry',
    'SerializedError',
    'PerformanceMetrics',
    'create_logger'
]