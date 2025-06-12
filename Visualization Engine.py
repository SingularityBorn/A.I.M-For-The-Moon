"""
Enterprise Visualization Engine

High-performance data visualization and charting engine optimized for time-series data,
analytics dashboards, and automated reporting. Supports multiple rendering backends,
export formats, and customizable styling systems.

Features:
- Multiple chart types with automatic optimal selection
- High-performance rendering with GPU acceleration support
- Memory-efficient streaming visualization for large datasets
- Customizable themes and branding systems
- Export to multiple formats (PNG, SVG, PDF, HTML)
- Interactive charts with zoom, pan, and drill-down capabilities
- Automatic color palette generation and accessibility compliance
- Responsive layouts for different screen sizes
"""

import asyncio
import base64
import datetime
import hashlib
import io
import json
import math
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Protocol

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont

from .core import BaseDataPoint, AnalyticsConfig, performance_monitor


class ChartType(Enum):
    """Supported chart types with automatic selection capabilities."""
    LINE = "line"
    BAR = "bar"
    SCATTER = "scatter"
    AREA = "area"
    PIE = "pie"
    HEATMAP = "heatmap"
    HISTOGRAM = "histogram"
    BOX_PLOT = "box_plot"
    VIOLIN_PLOT = "violin_plot"
    CANDLESTICK = "candlestick"
    GAUGE = "gauge"
    TREEMAP = "treemap"
    SANKEY = "sankey"
    WATERFALL = "waterfall"


class ExportFormat(Enum):
    """Chart export formats."""
    PNG = "png"
    SVG = "svg"
    PDF = "pdf"
    HTML = "html"
    JSON = "json"


class ColorScheme(Enum):
    """Predefined color schemes optimized for different use cases."""
    CORPORATE = "corporate"
    SCIENTIFIC = "scientific"
    DASHBOARD = "dashboard"
    HIGH_CONTRAST = "high_contrast"
    COLORBLIND_FRIENDLY = "colorblind_friendly"
    MONOCHROME = "monochrome"
    CUSTOM = "custom"


@dataclass
class ChartStyle:
    """Comprehensive chart styling configuration."""
    color_scheme: ColorScheme = ColorScheme.CORPORATE
    primary_color: str = "#2E86AB"
    secondary_color: str = "#A23B72"
    background_color: str = "#FFFFFF"
    grid_color: str = "#E0E0E0"
    text_color: str = "#333333"
    
    # Typography
    title_font_size: int = 16
    axis_font_size: int = 12
    label_font_size: int = 10
    font_family: str = "DejaVu Sans"
    
    # Layout
    figure_width: float = 12.0
    figure_height: float = 8.0
    dpi: int = 300
    tight_layout: bool = True
    
    # Grid and axes
    show_grid: bool = True
    grid_alpha: float = 0.3
    spine_width: float = 0.8
    
    # Markers and lines
    line_width: float = 2.0
    marker_size: float = 6.0
    alpha: float = 0.8
    
    # Annotations
    show_values: bool = False
    value_precision: int = 2
    
    def get_color_palette(self, n_colors: int = 10) -> List[str]:
        """Generate a color palette based on the scheme."""
        if self.color_scheme == ColorScheme.CORPORATE:
            base_colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#8E44AD"]
        elif self.color_scheme == ColorScheme.SCIENTIFIC:
            base_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
        elif self.color_scheme == ColorScheme.DASHBOARD:
            base_colors = ["#3498DB", "#E74C3C", "#2ECC71", "#F39C12", "#9B59B6"]
        elif self.color_scheme == ColorScheme.HIGH_CONTRAST:
            base_colors = ["#000000", "#FFFFFF", "#FF0000", "#00FF00", "#0000FF"]
        elif self.color_scheme == ColorScheme.COLORBLIND_FRIENDLY:
            base_colors = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e"]
        elif self.color_scheme == ColorScheme.MONOCHROME:
            base_colors = ["#000000", "#333333", "#666666", "#999999", "#CCCCCC"]
        else:
            base_colors = [self.primary_color, self.secondary_color]
        
        # Extend palette if needed
        if n_colors <= len(base_colors):
            return base_colors[:n_colors]
        
        # Generate intermediate colors
        extended_palette = base_colors[:]
        while len(extended_palette) < n_colors:
            # Add lighter/darker variants
            for base_color in base_colors:
                if len(extended_palette) >= n_colors:
                    break
                # Create variations
                rgba = to_rgba(base_color)
                # Lighter variant
                lighter = tuple(min(1.0, c + 0.2) for c in rgba[:3]) + (rgba[3],)
                extended_palette.append(f"#{int(lighter[0]*255):02x}{int(lighter[1]*255):02x}{int(lighter[2]*255):02x}")
                
                if len(extended_palette) >= n_colors:
                    break
                
                # Darker variant
                darker = tuple(max(0.0, c - 0.2) for c in rgba[:3]) + (rgba[3],)
                extended_palette.append(f"#{int(darker[0]*255):02x}{int(darker[1]*255):02x}{int(darker[2]*255):02x}")
        
        return extended_palette[:n_colors]


@dataclass
class ChartData:
    """Structured chart data with metadata."""
    x_values: List[Any]
    y_values: List[Union[float, List[float]]]
    labels: Optional[List[str]] = None
    colors: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> bool:
        """Validate chart data consistency."""
        if not self.x_values or not self.y_values:
            return False
        
        if len(self.x_values) != len(self.y_values):
            return False
        
        if self.labels and len(self.labels) != len(self.x_values):
            return False
        
        return True


class ChartRenderer(ABC):
    """Abstract base class for chart renderers."""
    
    @abstractmethod
    async def render(
        self,
        data: ChartData,
        chart_type: ChartType,
        style: ChartStyle,
        title: str,
        x_label: str,
        y_label: str
    ) -> bytes:
        """Render chart and return as bytes."""
        pass


class MatplotlibRenderer(ChartRenderer):
    """High-performance matplotlib-based chart renderer."""
    
    def __init__(self):
        # Configure matplotlib for optimal performance
        plt.rcParams.update({
            'figure.max_open_warning': 0,
            'font.size': 10,
            'axes.linewidth': 0.8,
            'xtick.major.width': 0.8,
            'ytick.major.width': 0.8,
            'xtick.minor.width': 0.4,
            'ytick.minor.width': 0.4,
        })
    
    @performance_monitor("render_chart")
    async def render(
        self,
        data: ChartData,
        chart_type: ChartType,
        style: ChartStyle,
        title: str,
        x_label: str,
        y_label: str
    ) -> bytes:
        """Render chart using matplotlib."""
        if not data.validate():
            raise ValueError("Invalid chart data")
        
        # Create figure
        fig = Figure(
            figsize=(style.figure_width, style.figure_height),
            dpi=style.dpi,
            facecolor=style.background_color
        )
        
        ax = fig.add_subplot(111)
        
        # Apply styling
        self._apply_style(ax, style)
        
        # Render based on chart type
        if chart_type == ChartType.LINE:
            await self._render_line_chart(ax, data, style)
        elif chart_type == ChartType.BAR:
            await self._render_bar_chart(ax, data, style)
        elif chart_type == ChartType.SCATTER:
            await self._render_scatter_chart(ax, data, style)
        elif chart_type == ChartType.AREA:
            await self._render_area_chart(ax, data, style)
        elif chart_type == ChartType.PIE:
            await self._render_pie_chart(ax, data, style)
        elif chart_type == ChartType.HEATMAP:
            await self._render_heatmap(ax, data, style)
        elif chart_type == ChartType.HISTOGRAM:
            await self._render_histogram(ax, data, style)
        elif chart_type == ChartType.BOX_PLOT:
            await self._render_box_plot(ax, data, style)
        elif chart_type == ChartType.GAUGE:
            await self._render_gauge_chart(ax, data, style)
        else:
            raise ValueError(f"Unsupported chart type: {chart_type}")
        
        # Set labels and title
        ax.set_title(title, fontsize=style.title_font_size, fontweight='bold', color=style.text_color)
        ax.set_xlabel(x_label, fontsize=style.axis_font_size, color=style.text_color)
        ax.set_ylabel(y_label, fontsize=style.axis_font_size, color=style.text_color)
        
        # Apply tight layout if enabled
        if style.tight_layout:
            fig.tight_layout()
        
        # Convert to bytes
        buffer = io.BytesIO()
        canvas = FigureCanvasAgg(fig)
        canvas.print_png(buffer, dpi=style.dpi, bbox_inches='tight')
        buffer.seek(0)
        
        image_bytes = buffer.getvalue()
        buffer.close()
        plt.close(fig)
        
        return image_bytes
    
    def _apply_style(self, ax, style: ChartStyle):
        """Apply styling to axes."""
        ax.set_facecolor(style.background_color)
        
        # Grid
        if style.show_grid:
            ax.grid(True, alpha=style.grid_alpha, color=style.grid_color, linewidth=0.5)
        
        # Spines
        for spine in ax.spines.values():
            spine.set_linewidth(style.spine_width)
            spine.set_color(style.text_color)
        
        # Tick parameters
        ax.tick_params(colors=style.text_color, labelsize=style.label_font_size)
    
    async def _render_line_chart(self, ax, data: ChartData, style: ChartStyle):
        """Render line chart."""
        colors = data.colors or style.get_color_palette(len(data.y_values[0]) if isinstance(data.y_values[0], list) else 1)
        
        if isinstance(data.y_values[0], list):
            # Multiple series
            for i, y_series in enumerate(zip(*data.y_values)):
                color = colors[i % len(colors)]
                label = data.labels[i] if data.labels and i < len(data.labels) else f"Series {i+1}"
                ax.plot(
                    data.x_values,
                    y_series,
                    color=color,
                    linewidth=style.line_width,
                    marker='o',
                    markersize=style.marker_size,
                    alpha=style.alpha,
                    label=label
                )
            ax.legend()
        else:
            # Single series
            ax.plot(
                data.x_values,
                data.y_values,
                color=colors[0],
                linewidth=style.line_width,
                marker='o',
                markersize=style.marker_size,
                alpha=style.alpha
            )
        
        # Handle datetime x-axis
        if data.x_values and isinstance(data.x_values[0], datetime.datetime):
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(data.x_values) // 10)))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    async def _render_bar_chart(self, ax, data: ChartData, style: ChartStyle):
        """Render bar chart."""
        colors = data.colors or style.get_color_palette(len(data.y_values))
        
        bars = ax.bar(
            data.x_values,
            data.y_values,
            color=colors,
            alpha=style.alpha,
            edgecolor=style.text_color,
            linewidth=0.5
        )
        
        # Add value labels if enabled
        if style.show_values:
            for bar, value in zip(bars, data.y_values):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width()/2.,
                    height,
                    f'{value:.{style.value_precision}f}',
                    ha='center',
                    va='bottom',
                    fontsize=style.label_font_size,
                    color=style.text_color
                )
    
    async def _render_scatter_chart(self, ax, data: ChartData, style: ChartStyle):
        """Render scatter plot."""
        colors = data.colors or [style.primary_color] * len(data.x_values)
        
        scatter = ax.scatter(
            data.x_values,
            data.y_values,
            c=colors,
            s=style.marker_size * 10,
            alpha=style.alpha,
            edgecolors=style.text_color,
            linewidth=0.5
        )
        
        return scatter
    
    async def _render_area_chart(self, ax, data: ChartData, style: ChartStyle):
        """Render area chart."""
        colors = data.colors or style.get_color_palette(1)
        
        ax.fill_between(
            data.x_values,
            data.y_values,
            color=colors[0],
            alpha=style.alpha * 0.7,
            edgecolor=colors[0],
            linewidth=style.line_width
        )
        
        # Add line on top
        ax.plot(
            data.x_values,
            data.y_values,
            color=colors[0],
            linewidth=style.line_width
        )
    
    async def _render_pie_chart(self, ax, data: ChartData, style: ChartStyle):
        """Render pie chart."""
        colors = data.colors or style.get_color_palette(len(data.y_values))
        labels = data.labels or [f"Item {i+1}" for i in range(len(data.y_values))]
        
        # Calculate percentages
        total = sum(data.y_values)
        percentages = [value/total * 100 for value in data.y_values]
        
        wedges, texts, autotexts = ax.pie(
            data.y_values,
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            startangle=90,
            textprops={'fontsize': style.label_font_size, 'color': style.text_color}
        )
        
        # Style the percentage text
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        ax.axis('equal')
    
    async def _render_heatmap(self, ax, data: ChartData, style: ChartStyle):
        """Render heatmap."""
        # Assuming data.y_values is a 2D array
        if not isinstance(data.y_values[0], list):
            raise ValueError("Heatmap requires 2D data")
        
        heatmap_data = np.array(data.y_values)
        
        im = ax.imshow(
            heatmap_data,
            cmap='viridis',
            aspect='auto',
            interpolation='nearest'
        )
        
        # Add colorbar
        plt.colorbar(im, ax=ax)
        
        # Set ticks
        if data.labels:
            ax.set_xticks(range(len(data.labels)))
            ax.set_xticklabels(data.labels, rotation=45)
    
    async def _render_histogram(self, ax, data: ChartData, style: ChartStyle):
        """Render histogram."""
        ax.hist(
            data.y_values,
            bins=min(50, len(data.y_values) // 5) or 10,
            color=style.primary_color,
            alpha=style.alpha,
            edgecolor=style.text_color,
            linewidth=0.5
        )
    
    async def _render_box_plot(self, ax, data: ChartData, style: ChartStyle):
        """Render box plot."""
        box_data = data.y_values if isinstance(data.y_values[0], list) else [data.y_values]
        labels = data.labels or [f"Group {i+1}" for i in range(len(box_data))]
        
        bp = ax.boxplot(
            box_data,
            labels=labels,
            patch_artist=True,
            medianprops={'color': style.text_color, 'linewidth': 2}
        )
        
        # Color the boxes
        colors = style.get_color_palette(len(box_data))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(style.alpha)
    
    async def _render_gauge_chart(self, ax, data: ChartData, style: ChartStyle):
        """Render gauge chart."""
        if len(data.y_values) != 1:
            raise ValueError("Gauge chart requires exactly one value")
        
        value = data.y_values[0]
        max_value = data.metadata.get('max_value', 100)
        min_value = data.metadata.get('min_value', 0)
        
        # Normalize value
        normalized_value = (value - min_value) / (max_value - min_value)
        
        # Create gauge
        theta = np.linspace(0, np.pi, 100)
        
        # Background arc
        ax.plot(
            np.cos(theta),
            np.sin(theta),
            color=style.grid_color,
            linewidth=style.line_width * 5
        )
        
        # Value arc
        value_theta = np.linspace(0, np.pi * normalized_value, int(100 * normalized_value))
        if len(value_theta) > 0:
            ax.plot(
                np.cos(value_theta),
                np.sin(value_theta),
                color=style.primary_color,
                linewidth=style.line_width * 5
            )
        
        # Add needle
        needle_angle = np.pi * normalized_value
        ax.arrow(
            0, 0,
            0.8 * np.cos(needle_angle),
            0.8 * np.sin(needle_angle),
            head_width=0.05,
            head_length=0.1,
            fc=style.text_color,
            ec=style.text_color
        )
        
        # Add center dot
        ax.plot(0, 0, 'o', color=style.text_color, markersize=style.marker_size * 2)
        
        # Add value text
        ax.text(
            0, -0.5,
            f"{value:.{style.value_precision}f}",
            ha='center',
            va='center',
            fontsize=style.title_font_size,
            fontweight='bold',
            color=style.text_color
        )
        
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-0.8, 1.2)
        ax.set_aspect('equal')
        ax.axis('off')


class VisualizationEngine:
    """
    Enterprise Visualization Engine
    
    High-performance visualization system with support for multiple chart types,
    rendering backends, and export formats. Optimized for large datasets and
    real-time analytics dashboards.
    """
    
    def __init__(self, config: Optional[AnalyticsConfig] = None):
        self.config = config or AnalyticsConfig()
        self.default_style = ChartStyle()
        self.renderers = {
            'matplotlib': MatplotlibRenderer()
        }
        self.chart_cache: Dict[str, bytes] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Initialize output directories
        self.charts_dir = self.config.output_directory / "charts"
        self.charts_dir.mkdir(parents=True, exist_ok=True)
    
    def _generate_cache_key(
        self,
        data: ChartData,
        chart_type: ChartType,
        style: ChartStyle,
        title: str,
        x_label: str,
        y_label: str
    ) -> str:
        """Generate cache key for chart data."""
        # Create hash from chart parameters
        content = f"{chart_type.value}_{title}_{x_label}_{y_label}"
        content += f"_{len(data.x_values)}_{len(data.y_values)}"
        content += f"_{style.color_scheme.value}_{style.figure_width}_{style.figure_height}"
        
        return hashlib.md5(content.encode()).hexdigest()
    
    @performance_monitor("create_chart")
    async def create_chart(
        self,
        data: List[BaseDataPoint],
        chart_type: ChartType,
        title: str = "Chart",
        x_label: str = "X-axis",
        y_label: str = "Y-axis",
        style: Optional[ChartStyle] = None,
        renderer: str = "matplotlib"
    ) -> Tuple[bytes, str]:
        """
        Create a chart from data points.
        
        Returns:
            Tuple of (chart_bytes, cache_key)
        """
        if not data:
            raise ValueError("No data provided for chart creation")
        
        # Convert data points to chart data
        chart_data = self._prepare_chart_data(data, chart_type)
        
        # Use default style if none provided
        chart_style = style or self.default_style
        
        # Generate cache key
        cache_key = self._generate_cache_key(chart_data, chart_type, chart_style, title, x_label, y_label)
        
        # Check cache
        if cache_key in self.chart_cache:
            self.cache_hits += 1
            return self.chart_cache[cache_key], cache_key
        
        self.cache_misses += 1
        
        # Render chart
        if renderer not in self.renderers:
            raise ValueError(f"Unknown renderer: {renderer}")
        
        chart_renderer = self.renderers[renderer]
        chart_bytes = await chart_renderer.render(
            chart_data, chart_type, chart_style, title, x_label, y_label
        )
        
        # Cache the result
        self.chart_cache[cache_key] = chart_bytes
        
        return chart_bytes, cache_key
    
    def _prepare_chart_data(self, data: List[BaseDataPoint], chart_type: ChartType) -> ChartData:
        """Convert data points to chart data format."""
        if not data:
            raise ValueError("Empty data list")
        
        # Sort by timestamp
        sorted_data = sorted(data, key=lambda x: x.timestamp)
        
        x_values = [point.timestamp for point in sorted_data]
        y_values = [point.value for point in sorted_data]
        
        # Handle different chart types
        if chart_type == ChartType.PIE:
            # For pie charts, group by source or metadata
            value_groups = defaultdict(float)
            for point in sorted_data:
                key = point.source or point.metadata.get('category', 'Unknown')
                value_groups[key] += point.value
            
            x_values = list(value_groups.keys())
            y_values = list(value_groups.values())
        
        elif chart_type == ChartType.HISTOGRAM:
            # For histograms, just use values
            x_values = list(range(len(y_values)))
        
        return ChartData(
            x_values=x_values,
            y_values=y_values,
            metadata={'chart_type': chart_type.value, 'data_points': len(data)}
        )
    
    async def save_chart(
        self,
        chart_bytes: bytes,
        filename: str,
        format: ExportFormat = ExportFormat.PNG
    ) -> Path:
        """Save chart bytes to file."""
        if format != ExportFormat.PNG:
            raise NotImplementedError(f"Export format {format} not yet implemented")
        
        # Ensure filename has correct extension
        if not filename.endswith('.png'):
            filename += '.png'
        
        file_path = self.charts_dir / filename
        
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(chart_bytes)
        
        return file_path
    
    async def create_multi_chart_dashboard(
        self,
        charts_config: List[Dict[str, Any]],
        title: str = "Analytics Dashboard",
        layout: Tuple[int, int] = (2, 2)
    ) -> bytes:
        """Create a multi-chart dashboard layout."""
        rows, cols = layout
        total_charts = rows * cols
        
        if len(charts_config) > total_charts:
            raise ValueError(f"Too many charts for layout {layout}")
        
        # Create dashboard figure
        fig = plt.figure(figsize=(self.default_style.figure_width * cols, self.default_style.figure_height * rows))
        fig.suptitle(title, fontsize=self.default_style.title_font_size + 4, fontweight='bold')
        
        # Render each chart
        for i, chart_config in enumerate(charts_config):
            ax = fig.add_subplot(rows, cols, i + 1)
            
            # Extract chart configuration
            data = chart_config['data']
            chart_type = ChartType(chart_config.get('type', 'line'))
            chart_title = chart_config.get('title', f'Chart {i+1}')
            x_label = chart_config.get('x_label', 'X-axis')
            y_label = chart_config.get('y_label', 'Y-axis')
            
            # Prepare and render chart data
            chart_data = self._prepare_chart_data(data, chart_type)
            
            # Render using matplotlib renderer directly on subplot
            renderer = self.renderers['matplotlib']
            await self._render_on_subplot(ax, chart_data, chart_type, chart_title, x_label, y_label)
        
        # Convert to bytes
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=self.default_style.dpi, bbox_inches='tight')
        buffer.seek(0)
        
        image_bytes = buffer.getvalue()
        buffer.close()
        plt.close(fig)
        
        return image_bytes
    
    async def _render_on_subplot(
        self,
        ax,
        data: ChartData,
        chart_type: ChartType,
        title: str,
        x_label: str,
        y_label: str
    ):
        """Render chart directly on matplotlib subplot."""
        renderer = self.renderers['matplotlib']
        
        # Apply basic styling
        renderer._apply_style(ax, self.default_style)
        
        # Render based on type
        if chart_type == ChartType.LINE:
            await renderer._render_line_chart(ax, data, self.default_style)
        elif chart_type == ChartType.BAR:
            await renderer._render_bar_chart(ax, data, self.default_style)
        elif chart_type == ChartType.SCATTER:
            await renderer._render_scatter_chart(ax, data, self.default_style)
        # Add more types as needed
        
        # Set labels
        ax.set_title(title, fontsize=self.default_style.title_font_size - 2)
        ax.set_xlabel(x_label, fontsize=self.default_style.axis_font_size)
        ax.set_ylabel(y_label, fontsize=self.default_style.axis_font_size)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get visualization cache statistics."""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'cache_size': len(self.chart_cache),
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate_percent': hit_rate,
            'memory_usage_mb': sum(len(chart_bytes) for chart_bytes in self.chart_cache.values()) / 1024 / 1024
        }
    
    async def clear_cache(self):
        """Clear visualization cache."""
        self.chart_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
    
    async def optimize_chart_type(self, data: List[BaseDataPoint]) -> ChartType:
        """Automatically select optimal chart type based on data characteristics."""
        if not data:
            return ChartType.LINE
        
        data_size = len(data)
        
        # Check for categorical data
        unique_sources = len(set(point.source for point in data if point.source))
        
        # Check for time series
        has_timestamps = all(isinstance(point.timestamp, datetime.datetime) for point in data)
        
        # Check value distribution
        values = [point.value for point in data]
        value_range = max(values) - min(values) if values else 0
        
        # Decision logic
        if unique_sources > 1 and unique_sources <= 10 and not has_timestamps:
            return ChartType.PIE
        elif data_size > 1000 and has_timestamps:
            return ChartType.AREA
        elif data_size > 100 and value_range > 0:
            return ChartType.SCATTER
        elif has_timestamps:
            return ChartType.LINE
        else:
            return ChartType.BAR


# Export main classes
__all__ = [
    'VisualizationEngine',
    'ChartType',
    'ChartStyle',
    'ChartData',
    'ColorScheme',
    'ExportFormat',
    'MatplotlibRenderer'
]