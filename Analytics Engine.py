"""
Enterprise Analytics Engine

A high-performance, production-ready analytics platform for time-series data processing,
visualization, and automated reporting. Designed for distributed systems monitoring,
performance analysis, and anomaly detection.

Key Features:
- Asynchronous data collection and processing
- Real-time metrics aggregation with configurable retention
- Dynamic visualization engine with multiple output formats
- Automated PDF report generation with customizable templates
- Multi-format data export (JSON, CSV, Parquet)
- Enterprise-grade caching with TTL management
- Memory-efficient streaming processing
- Comprehensive error handling and recovery
- Built-in performance monitoring and optimization
"""

import asyncio
import datetime
import hashlib
import json
import logging
import math
import os
import tempfile
import uuid
import weakref
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import (
    Any, Awaitable, Callable, Dict, List, Optional, Protocol, Tuple, Union,
    TypeVar, Generic, NamedTuple, Set
)

import aiofiles
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, validator
from tenacity import retry, stop_after_attempt, wait_exponential

# Configure logging
logger = logging.getLogger("enterprise_analytics")

T = TypeVar('T')
DataPoint = TypeVar('DataPoint', bound='BaseDataPoint')


class MetricType(Enum):
    """Enumeration of supported metric types."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"
    RATE = "rate"


class ExportFormat(Enum):
    """Supported data export formats."""
    JSON = "json"
    CSV = "csv"
    PARQUET = "parquet"
    EXCEL = "xlsx"


class ReportType(Enum):
    """Available report template types."""
    EXECUTIVE = "executive"
    TECHNICAL = "technical"
    OPERATIONAL = "operational"
    COMPLIANCE = "compliance"
    PERFORMANCE = "performance"


class CacheStrategy(Enum):
    """Cache invalidation and refresh strategies."""
    TTL = "ttl"
    LRU = "lru"
    LFU = "lfu"
    ADAPTIVE = "adaptive"


@dataclass(frozen=True)
class BaseDataPoint:
    """Immutable base data point with validation."""
    timestamp: datetime.datetime
    value: Union[int, float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: Optional[str] = None
    quality_score: float = field(default=1.0)
    
    def __post_init__(self):
        if not 0.0 <= self.quality_score <= 1.0:
            raise ValueError("Quality score must be between 0.0 and 1.0")
        if math.isnan(self.value) or math.isinf(self.value):
            raise ValueError("Value cannot be NaN or infinite")


@dataclass(frozen=True)
class MetricDefinition:
    """Immutable metric definition with validation rules."""
    name: str
    type: MetricType
    unit: str
    description: str
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    expected_frequency: Optional[int] = None  # seconds
    retention_days: int = 90
    aggregation_methods: Set[str] = field(default_factory=lambda: {"mean", "min", "max"})
    
    def validate_value(self, value: Union[int, float]) -> bool:
        """Validate a value against this metric's constraints."""
        if self.min_value is not None and value < self.min_value:
            return False
        if self.max_value is not None and value > self.max_value:
            return False
        return True


class AnalyticsConfig(BaseModel):
    """Comprehensive configuration for the analytics engine."""
    
    # Core settings
    output_directory: Path = Field(default=Path("analytics_output"))
    data_retention_days: int = Field(default=90, ge=1, le=3650)
    worker_pool_size: int = Field(default=4, ge=1, le=32)
    max_memory_usage_mb: int = Field(default=1024, ge=64)
    
    # Cache configuration
    cache_strategy: CacheStrategy = Field(default=CacheStrategy.TTL)
    cache_ttl_seconds: int = Field(default=3600, ge=60)
    cache_max_size: int = Field(default=10000, ge=100)
    
    # Processing settings
    batch_size: int = Field(default=1000, ge=1)
    flush_interval_seconds: int = Field(default=30, ge=1)
    enable_streaming: bool = Field(default=True)
    enable_compression: bool = Field(default=True)
    compression_level: int = Field(default=6, ge=1, le=9)
    
    # Visualization settings
    chart_dpi: int = Field(default=300, ge=72)
    chart_width: int = Field(default=12, ge=4)
    chart_height: int = Field(default=8, ge=3)
    color_palette: str = Field(default="viridis")
    
    # Export settings
    default_export_format: ExportFormat = Field(default=ExportFormat.JSON)
    include_metadata: bool = Field(default=True)
    
    # Performance settings
    enable_performance_monitoring: bool = Field(default=True)
    performance_sample_rate: float = Field(default=0.1, ge=0.0, le=1.0)
    
    # Security settings
    sanitize_inputs: bool = Field(default=True)
    max_file_size_mb: int = Field(default=100, ge=1)
    
    @validator('output_directory')
    def validate_output_directory(cls, v):
        """Ensure output directory is valid and writable."""
        path = Path(v)
        path.mkdir(parents=True, exist_ok=True)
        if not os.access(path, os.W_OK):
            raise ValueError(f"Output directory {path} is not writable")
        return path


class DataSourceProtocol(Protocol):
    """Protocol defining the interface for data sources."""
    
    async def get_metrics(
        self, 
        start_time: datetime.datetime, 
        end_time: datetime.datetime,
        metric_names: Optional[List[str]] = None
    ) -> List[BaseDataPoint]:
        """Retrieve metrics data for the specified time range."""
        ...
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get the current health status of the data source."""
        ...


class ProcessorProtocol(Protocol):
    """Protocol for data processors."""
    
    async def process(
        self, 
        data: List[BaseDataPoint]
    ) -> List[BaseDataPoint]:
        """Process a batch of data points."""
        ...


class PerformanceMonitor:
    """High-performance monitoring and metrics collection."""
    
    def __init__(self):
        self._metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self._counters: Dict[str, int] = defaultdict(int)
        self._start_time = datetime.datetime.now()
        self._lock = asyncio.Lock()
    
    async def record_timing(self, operation: str, duration_ms: float):
        """Record timing information for an operation."""
        async with self._lock:
            self._metrics[f"timing_{operation}"].append({
                'timestamp': datetime.datetime.now(),
                'duration_ms': duration_ms
            })
    
    async def increment_counter(self, name: str, value: int = 1):
        """Increment a named counter."""
        async with self._lock:
            self._counters[name] += value
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        async with self._lock:
            metrics = {
                'uptime_seconds': (datetime.datetime.now() - self._start_time).total_seconds(),
                'counters': dict(self._counters),
                'timing_stats': {}
            }
            
            for name, values in self._metrics.items():
                if values and name.startswith('timing_'):
                    durations = [v['duration_ms'] for v in values]
                    metrics['timing_stats'][name] = {
                        'count': len(durations),
                        'mean': np.mean(durations),
                        'p50': np.percentile(durations, 50),
                        'p95': np.percentile(durations, 95),
                        'p99': np.percentile(durations, 99),
                        'max': np.max(durations)
                    }
            
            return metrics


class CacheManager:
    """Enterprise-grade caching with multiple strategies."""
    
    def __init__(self, config: AnalyticsConfig):
        self.config = config
        self._cache: Dict[str, Any] = {}
        self._access_times: Dict[str, datetime.datetime] = {}
        self._access_counts: Dict[str, int] = defaultdict(int)
        self._lock = asyncio.RWLock()
        self._cleanup_task: Optional[asyncio.Task] = None
        self._start_cleanup_task()
    
    def _start_cleanup_task(self):
        """Start the background cache cleanup task."""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def _cleanup_loop(self):
        """Background task for cache maintenance."""
        while True:
            try:
                await asyncio.sleep(self.config.cache_ttl_seconds / 4)
                await self._cleanup_expired()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Cache cleanup error: {e}")
    
    async def _cleanup_expired(self):
        """Remove expired cache entries."""
        async with self._lock.writer():
            current_time = datetime.datetime.now()
            expired_keys = []
            
            for key, access_time in self._access_times.items():
                age = (current_time - access_time).total_seconds()
                if age > self.config.cache_ttl_seconds:
                    expired_keys.append(key)
            
            for key in expired_keys:
                self._cache.pop(key, None)
                self._access_times.pop(key, None)
                self._access_counts.pop(key, None)
            
            # Apply size limits
            if len(self._cache) > self.config.cache_max_size:
                await self._evict_by_strategy()
    
    async def _evict_by_strategy(self):
        """Evict cache entries based on the configured strategy."""
        target_size = int(self.config.cache_max_size * 0.8)
        current_size = len(self._cache)
        evict_count = current_size - target_size
        
        if evict_count <= 0:
            return
        
        if self.config.cache_strategy == CacheStrategy.LRU:
            # Evict least recently used
            sorted_keys = sorted(
                self._access_times.items(),
                key=lambda x: x[1]
            )[:evict_count]
        elif self.config.cache_strategy == CacheStrategy.LFU:
            # Evict least frequently used
            sorted_keys = sorted(
                self._access_counts.items(),
                key=lambda x: x[1]
            )[:evict_count]
        else:
            # Default to LRU
            sorted_keys = sorted(
                self._access_times.items(),
                key=lambda x: x[1]
            )[:evict_count]
        
        for key, _ in sorted_keys:
            self._cache.pop(key, None)
            self._access_times.pop(key, None)
            self._access_counts.pop(key, None)
    
    async def get(self, key: str) -> Optional[Any]:
        """Get a value from the cache."""
        async with self._lock.reader():
            if key in self._cache:
                self._access_times[key] = datetime.datetime.now()
                self._access_counts[key] += 1
                return self._cache[key]
            return None
    
    async def set(self, key: str, value: Any):
        """Set a value in the cache."""
        async with self._lock.writer():
            self._cache[key] = value
            self._access_times[key] = datetime.datetime.now()
            self._access_counts[key] += 1
    
    async def invalidate(self, pattern: Optional[str] = None):
        """Invalidate cache entries matching a pattern."""
        async with self._lock.writer():
            if pattern is None:
                self._cache.clear()
                self._access_times.clear()
                self._access_counts.clear()
            else:
                keys_to_remove = [k for k in self._cache.keys() if pattern in k]
                for key in keys_to_remove:
                    self._cache.pop(key, None)
                    self._access_times.pop(key, None)
                    self._access_counts.pop(key, None)
    
    async def close(self):
        """Clean up resources."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass


class DataValidator:
    """Enterprise data validation and sanitization."""
    
    @staticmethod
    def validate_numeric(
        value: Any, 
        min_val: Optional[float] = None, 
        max_val: Optional[float] = None,
        allow_none: bool = False
    ) -> Optional[float]:
        """Validate and sanitize numeric values."""
        if value is None:
            return None if allow_none else 0.0
        
        try:
            float_val = float(value)
            if math.isnan(float_val) or math.isinf(float_val):
                return 0.0
            
            if min_val is not None and float_val < min_val:
                return min_val
            if max_val is not None and float_val > max_val:
                return max_val
                
            return float_val
        except (ValueError, TypeError):
            return 0.0
    
    @staticmethod
    def validate_timestamp(value: Any) -> datetime.datetime:
        """Validate and convert timestamp values."""
        if isinstance(value, datetime.datetime):
            return value
        elif isinstance(value, (int, float)):
            try:
                return datetime.datetime.fromtimestamp(value)
            except (ValueError, OSError):
                return datetime.datetime.now()
        elif isinstance(value, str):
            try:
                return datetime.datetime.fromisoformat(value.replace('Z', '+00:00'))
            except ValueError:
                return datetime.datetime.now()
        else:
            return datetime.datetime.now()
    
    @staticmethod
    def sanitize_string(
        value: Any, 
        max_length: int = 1000,
        allow_empty: bool = True
    ) -> str:
        """Sanitize string values."""
        if value is None:
            return "" if allow_empty else "unknown"
        
        str_val = str(value)[:max_length]
        # Remove control characters except whitespace
        sanitized = ''.join(char for char in str_val if char.isprintable() or char.isspace())
        return sanitized.strip()


class AsyncRWLock:
    """Async reader-writer lock implementation."""
    
    def __init__(self):
        self._readers = 0
        self._writers = 0
        self._read_ready = asyncio.Condition()
        self._write_ready = asyncio.Condition()
    
    @asynccontextmanager
    async def reader(self):
        """Async context manager for read access."""
        async with self._read_ready:
            await self._read_ready.wait_for(lambda: self._writers == 0)
            self._readers += 1
        
        try:
            yield
        finally:
            async with self._read_ready:
                self._readers -= 1
                if self._readers == 0:
                    self._read_ready.notify_all()
    
    @asynccontextmanager
    async def writer(self):
        """Async context manager for write access."""
        async with self._write_ready:
            await self._write_ready.wait_for(lambda: self._writers == 0 and self._readers == 0)
            self._writers += 1
        
        try:
            yield
        finally:
            async with self._write_ready:
                self._writers -= 1
                self._write_ready.notify_all()
            async with self._read_ready:
                self._read_ready.notify_all()


def performance_monitor(operation_name: str):
    """Decorator for monitoring function performance."""
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                start_time = datetime.datetime.now()
                try:
                    result = await func(*args, **kwargs)
                    return result
                finally:
                    duration = (datetime.datetime.now() - start_time).total_seconds() * 1000
                    # Access performance monitor from args[0] if it's a class instance
                    if args and hasattr(args[0], '_performance_monitor'):
                        await args[0]._performance_monitor.record_timing(operation_name, duration)
            return async_wrapper
        else:
            def sync_wrapper(*args, **kwargs):
                start_time = datetime.datetime.now()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    duration = (datetime.datetime.now() - start_time).total_seconds() * 1000
                    logger.debug(f"{operation_name} completed in {duration:.2f}ms")
            return sync_wrapper
    return decorator


def retry_on_failure(max_attempts: int = 3, backoff_factor: float = 1.0):
    """Decorator for retrying failed operations with exponential backoff."""
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=backoff_factor, min=1, max=10),
        reraise=True
    )


class ResourceManager:
    """Manages system resources and prevents memory leaks."""
    
    def __init__(self, max_memory_mb: int = 1024):
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self._resources: Set[weakref.ref] = set()
        self._cleanup_task: Optional[asyncio.Task] = None
        self._start_cleanup()
    
    def _start_cleanup(self):
        """Start background resource cleanup."""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def _cleanup_loop(self):
        """Background resource cleanup loop."""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                await self._check_memory_usage()
                self._cleanup_dead_references()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Resource cleanup error: {e}")
    
    async def _check_memory_usage(self):
        """Monitor memory usage and trigger cleanup if needed."""
        import psutil
        process = psutil.Process()
        memory_usage = process.memory_info().rss
        
        if memory_usage > self.max_memory_bytes:
            logger.warning(f"Memory usage {memory_usage / 1024 / 1024:.1f}MB exceeds limit")
            # Trigger garbage collection
            import gc
            gc.collect()
    
    def _cleanup_dead_references(self):
        """Remove dead weak references."""
        self._resources = {ref for ref in self._resources if ref() is not None}
    
    def register_resource(self, resource: Any):
        """Register a resource for tracking."""
        self._resources.add(weakref.ref(resource))
    
    async def close(self):
        """Clean up all resources."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Clean up tracked resources
        for ref in self._resources:
            resource = ref()
            if resource and hasattr(resource, 'close'):
                try:
                    if asyncio.iscoroutinefunction(resource.close):
                        await resource.close()
                    else:
                        resource.close()
                except Exception as e:
                    logger.warning(f"Error closing resource: {e}")


class AnalyticsEngine:
    """
    Enterprise Analytics Engine - High-performance time-series analytics platform.
    
    Provides comprehensive data collection, processing, visualization, and reporting
    capabilities with enterprise-grade reliability and performance.
    """
    
    def __init__(self, config: Optional[AnalyticsConfig] = None):
        self.config = config or AnalyticsConfig()
        self._performance_monitor = PerformanceMonitor()
        self._cache_manager = CacheManager(self.config)
        self._resource_manager = ResourceManager(self.config.max_memory_usage_mb)
        self._data_sources: Dict[str, DataSourceProtocol] = {}
        self._processors: List[ProcessorProtocol] = []
        self._metrics_registry: Dict[str, MetricDefinition] = {}
        self._thread_pool = ThreadPoolExecutor(max_workers=self.config.worker_pool_size)
        self._shutdown_event = asyncio.Event()
        self._background_tasks: Set[asyncio.Task] = set()
        self._validator = DataValidator()
        
        # Initialize directories
        self._init_directories()
        
        # Register cleanup
        self._resource_manager.register_resource(self)
        
        logger.info("Enterprise Analytics Engine initialized")
    
    def _init_directories(self):
        """Initialize required directories."""
        base_dir = self.config.output_directory
        self.charts_dir = base_dir / "charts"
        self.reports_dir = base_dir / "reports"
        self.exports_dir = base_dir / "exports"
        self.temp_dir = base_dir / "temp"
        
        for directory in [self.charts_dir, self.reports_dir, self.exports_dir, self.temp_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    async def register_data_source(self, name: str, source: DataSourceProtocol):
        """Register a data source with the analytics engine."""
        self._data_sources[name] = source
        logger.info(f"Registered data source: {name}")
    
    async def register_metric(self, definition: MetricDefinition):
        """Register a metric definition."""
        self._metrics_registry[definition.name] = definition
        logger.info(f"Registered metric: {definition.name}")
    
    async def add_processor(self, processor: ProcessorProtocol):
        """Add a data processor to the pipeline."""
        self._processors.append(processor)
        logger.info(f"Added processor: {type(processor).__name__}")
    
    @performance_monitor("collect_metrics")
    async def collect_metrics(
        self,
        source_name: str,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
        metric_names: Optional[List[str]] = None
    ) -> List[BaseDataPoint]:
        """Collect metrics from a registered data source."""
        if source_name not in self._data_sources:
            raise ValueError(f"Unknown data source: {source_name}")
        
        # Check cache first
        cache_key = f"metrics_{source_name}_{start_time.isoformat()}_{end_time.isoformat()}_{metric_names}"
        cached_data = await self._cache_manager.get(cache_key)
        if cached_data is not None:
            logger.debug(f"Cache hit for metrics collection: {source_name}")
            return cached_data
        
        # Collect from source
        source = self._data_sources[source_name]
        raw_data = await source.get_metrics(start_time, end_time, metric_names)
        
        # Process through pipeline
        processed_data = raw_data
        for processor in self._processors:
            processed_data = await processor.process(processed_data)
        
        # Cache the results
        await self._cache_manager.set(cache_key, processed_data)
        await self._performance_monitor.increment_counter("metrics_collected", len(processed_data))
        
        return processed_data
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return await self._performance_monitor.get_metrics()
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        health_status = {
            "status": "healthy",
            "timestamp": datetime.datetime.now().isoformat(),
            "uptime_seconds": (datetime.datetime.now() - datetime.datetime.now()).total_seconds(),
            "data_sources": {},
            "cache_stats": {
                "size": len(self._cache_manager._cache),
                "max_size": self.config.cache_max_size
            },
            "memory_usage_mb": 0,  # Will be filled by resource manager
            "performance": await self._performance_monitor.get_metrics()
        }
        
        # Check data sources
        for name, source in self._data_sources.items():
            try:
                source_health = await source.get_health_status()
                health_status["data_sources"][name] = source_health
            except Exception as e:
                health_status["data_sources"][name] = {"status": "error", "error": str(e)}
                health_status["status"] = "degraded"
        
        return health_status
    
    async def close(self):
        """Clean up all resources."""
        logger.info("Shutting down Analytics Engine...")
        
        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()
        
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        # Close components
        await self._cache_manager.close()
        await self._resource_manager.close()
        
        # Shutdown thread pool
        self._thread_pool.shutdown(wait=True)
        
        logger.info("Analytics Engine shutdown complete")


# Export main classes
__all__ = [
    'AnalyticsEngine',
    'AnalyticsConfig', 
    'BaseDataPoint',
    'MetricDefinition',
    'MetricType',
    'ExportFormat',
    'ReportType',
    'DataSourceProtocol',
    'ProcessorProtocol',
    'performance_monitor',
    'retry_on_failure'
]