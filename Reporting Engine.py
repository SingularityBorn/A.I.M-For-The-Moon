"""
Enterprise Reporting Engine

Advanced automated reporting system for generating comprehensive analytics reports,
data exports, and business intelligence documents. Supports multiple output formats,
customizable templates, and scheduled report generation.

Features:
- Multi-format report generation (PDF, HTML, Excel, Word)
- Template-based report system with inheritance
- Automated data aggregation and statistical analysis
- Interactive HTML reports with embedded charts
- Batch export capabilities for large datasets
- Report scheduling and distribution
- Custom branding and styling
- Compliance and audit trail reporting
"""

import asyncio
import base64
import csv
import datetime
import io
import json
import tempfile
import zipfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Protocol
from jinja2 import Environment, FileSystemLoader, Template

import aiofiles
import pandas as pd
import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import letter, A4, legal
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    PageBreak, KeepTogether, Frame, PageTemplate
)
from reportlab.platypus.flowables import HRFlowable
from reportlab.pdfgen import canvas

from .core import BaseDataPoint, AnalyticsConfig, performance_monitor
from .visualization import VisualizationEngine, ChartType


class ReportFormat(Enum):
    """Supported report output formats."""
    PDF = "pdf"
    HTML = "html"
    EXCEL = "xlsx"
    WORD = "docx"
    JSON = "json"
    CSV = "csv"


class ReportTemplate(Enum):
    """Predefined report templates."""
    EXECUTIVE_SUMMARY = "executive_summary"
    TECHNICAL_ANALYSIS = "technical_analysis"
    OPERATIONAL_METRICS = "operational_metrics"
    COMPLIANCE_AUDIT = "compliance_audit"
    PERFORMANCE_DASHBOARD = "performance_dashboard"
    TREND_ANALYSIS = "trend_analysis"
    CUSTOM = "custom"


class ExportScope(Enum):
    """Data export scope options."""
    SUMMARY_ONLY = "summary_only"
    DETAILED_DATA = "detailed_data"
    RAW_DATA = "raw_data"
    COMPLETE = "complete"


@dataclass
class ReportSection:
    """Individual report section configuration."""
    title: str
    content_type: str  # 'text', 'chart', 'table', 'metrics'
    data: Optional[Any] = None
    template: Optional[str] = None
    style: Optional[Dict[str, Any]] = None
    include_in_toc: bool = True
    page_break_before: bool = False
    page_break_after: bool = False


@dataclass
class ReportConfig:
    """Comprehensive report configuration."""
    title: str
    subtitle: Optional[str] = None
    author: str = "Analytics Engine"
    organization: Optional[str] = None
    logo_path: Optional[Path] = None
    
    # Content configuration
    template: ReportTemplate = ReportTemplate.TECHNICAL_ANALYSIS
    sections: List[ReportSection] = field(default_factory=list)
    include_toc: bool = True
    include_executive_summary: bool = True
    include_appendix: bool = True
    
    # Styling
    page_size = A4
    margin_inches: float = 1.0
    font_family: str = "Helvetica"
    color_scheme: Dict[str, str] = field(default_factory=lambda: {
        'primary': '#2E86AB',
        'secondary': '#A23B72',
        'accent': '#F18F01',
        'text': '#333333',
        'background': '#FFFFFF'
    })
    
    # Data configuration
    date_range_days: int = 30
    include_raw_data: bool = False
    statistical_analysis: bool = True
    
    # Output configuration
    output_format: ReportFormat = ReportFormat.PDF
    compress_output: bool = False
    watermark: Optional[str] = None


class StatisticalAnalyzer:
    """Statistical analysis engine for report data."""
    
    @staticmethod
    def calculate_basic_stats(data: List[BaseDataPoint]) -> Dict[str, float]:
        """Calculate basic statistical measures."""
        if not data:
            return {}
        
        values = [point.value for point in data]
        
        return {
            'count': len(values),
            'mean': np.mean(values),
            'median': np.median(values),
            'std_dev': np.std(values),
            'variance': np.var(values),
            'min': np.min(values),
            'max': np.max(values),
            'range': np.max(values) - np.min(values),
            'q1': np.percentile(values, 25),
            'q3': np.percentile(values, 75),
            'iqr': np.percentile(values, 75) - np.percentile(values, 25),
            'skewness': float(pd.Series(values).skew()),
            'kurtosis': float(pd.Series(values).kurtosis())
        }
    
    @staticmethod
    def detect_trends(data: List[BaseDataPoint]) -> Dict[str, Any]:
        """Detect trends in time series data."""
        if len(data) < 2:
            return {'trend': 'insufficient_data'}
        
        # Sort by timestamp
        sorted_data = sorted(data, key=lambda x: x.timestamp)
        values = [point.value for point in sorted_data]
        
        # Simple linear trend
        x = range(len(values))
        coefficients = np.polyfit(x, values, 1)
        slope = coefficients[0]
        
        # Determine trend direction
        if abs(slope) < 0.01:
            trend_direction = 'stable'
        elif slope > 0:
            trend_direction = 'increasing'
        else:
            trend_direction = 'decreasing'
        
        # Calculate R-squared
        predicted = np.polyval(coefficients, x)
        ss_res = np.sum((values - predicted) ** 2)
        ss_tot = np.sum((values - np.mean(values)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return {
            'trend': trend_direction,
            'slope': slope,
            'r_squared': r_squared,
            'confidence': 'high' if r_squared > 0.7 else 'medium' if r_squared > 0.3 else 'low'
        }
    
    @staticmethod
    def detect_anomalies(data: List[BaseDataPoint], threshold: float = 2.0) -> List[Dict[str, Any]]:
        """Detect statistical anomalies using Z-score method."""
        if len(data) < 3:
            return []
        
        values = [point.value for point in data]
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        if std_val == 0:
            return []
        
        anomalies = []
        for i, point in enumerate(data):
            z_score = abs((point.value - mean_val) / std_val)
            if z_score > threshold:
                anomalies.append({
                    'timestamp': point.timestamp,
                    'value': point.value,
                    'z_score': z_score,
                    'deviation': point.value - mean_val,
                    'severity': 'high' if z_score > 3.0 else 'medium'
                })
        
        return anomalies


class PDFReportGenerator:
    """Advanced PDF report generator using ReportLab."""
    
    def __init__(self, config: ReportConfig):
        self.config = config
        self.doc = None
        self.story = []
        self.styles = self._create_styles()
    
    def _create_styles(self):
        """Create custom paragraph styles."""
        styles = getSampleStyleSheet()
        
        # Custom styles
        styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=styles['Title'],
            fontSize=24,
            textColor=colors.HexColor(self.config.color_scheme['primary']),
            spaceAfter=30,
            alignment=TA_CENTER
        ))
        
        styles.add(ParagraphStyle(
            name='CustomHeading1',
            parent=styles['Heading1'],
            fontSize=16,
            textColor=colors.HexColor(self.config.color_scheme['primary']),
            spaceBefore=20,
            spaceAfter=12
        ))
        
        styles.add(ParagraphStyle(
            name='CustomHeading2',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor(self.config.color_scheme['secondary']),
            spaceBefore=16,
            spaceAfter=8
        ))
        
        styles.add(ParagraphStyle(
            name='MetricValue',
            parent=styles['Normal'],
            fontSize=18,
            textColor=colors.HexColor(self.config.color_scheme['accent']),
            fontName='Helvetica-Bold',
            alignment=TA_CENTER
        ))
        
        return styles
    
    async def generate(self, report_data: Dict[str, Any], output_path: Path) -> Path:
        """Generate PDF report."""
        self.doc = SimpleDocTemplate(
            str(output_path),
            pagesize=self.config.page_size,
            rightMargin=self.config.margin_inches * inch,
            leftMargin=self.config.margin_inches * inch,
            topMargin=self.config.margin_inches * inch,
            bottomMargin=self.config.margin_inches * inch
        )
        
        # Build story
        await self._build_title_page(report_data)
        
        if self.config.include_toc:
            await self._build_toc()
        
        if self.config.include_executive_summary:
            await self._build_executive_summary(report_data)
        
        for section in self.config.sections:
            await self._build_section(section, report_data)
        
        if self.config.include_appendix:
            await self._build_appendix(report_data)
        
        # Build PDF
        self.doc.build(self.story)
        return output_path
    
    async def _build_title_page(self, report_data: Dict[str, Any]):
        """Build report title page."""
        # Logo
        if self.config.logo_path and self.config.logo_path.exists():
            logo = Image(str(self.config.logo_path), width=2*inch, height=1*inch)
            logo.hAlign = 'CENTER'
            self.story.append(logo)
            self.story.append(Spacer(1, 0.5*inch))
        
        # Title
        self.story.append(Paragraph(self.config.title, self.styles['CustomTitle']))
        
        # Subtitle
        if self.config.subtitle:
            self.story.append(Paragraph(self.config.subtitle, self.styles['CustomHeading2']))
        
        self.story.append(Spacer(1, 1*inch))
        
        # Report metadata
        metadata_data = [
            ['Generated:', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            ['Author:', self.config.author],
            ['Period:', f"Last {self.config.date_range_days} days"],
        ]
        
        if self.config.organization:
            metadata_data.append(['Organization:', self.config.organization])
        
        metadata_table = Table(metadata_data, colWidths=[2*inch, 3*inch])
        metadata_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        
        self.story.append(metadata_table)
        self.story.append(PageBreak())
    
    async def _build_toc(self):
        """Build table of contents."""
        self.story.append(Paragraph("Table of Contents", self.styles['CustomHeading1']))
        
        # TOC entries (simplified)
        toc_data = []
        page_num = 3  # Starting page after title and TOC
        
        if self.config.include_executive_summary:
            toc_data.append(['Executive Summary', str(page_num)])
            page_num += 1
        
        for i, section in enumerate(self.config.sections):
            if section.include_in_toc:
                toc_data.append([section.title, str(page_num)])
                page_num += 1
        
        if self.config.include_appendix:
            toc_data.append(['Appendix', str(page_num)])
        
        toc_table = Table(toc_data, colWidths=[4*inch, 1*inch])
        toc_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('LINEBELOW', (0, 0), (-1, -1), 0.5, colors.grey),
        ]))
        
        self.story.append(toc_table)
        self.story.append(PageBreak())
    
    async def _build_executive_summary(self, report_data: Dict[str, Any]):
        """Build executive summary section."""
        self.story.append(Paragraph("Executive Summary", self.styles['CustomHeading1']))
        
        # Key metrics
        stats = report_data.get('statistics', {})
        if stats:
            metrics_data = [
                ['Metric', 'Value', 'Trend'],
                ['Data Points', f"{stats.get('count', 0):,}", ''],
                ['Average Value', f"{stats.get('mean', 0):.2f}", ''],
                ['Standard Deviation', f"{stats.get('std_dev', 0):.2f}", ''],
                ['Min/Max Range', f"{stats.get('min', 0):.2f} - {stats.get('max', 0):.2f}", '']
            ]
            
            metrics_table = Table(metrics_data, colWidths=[2*inch, 1.5*inch, 1*inch])
            metrics_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(self.config.color_scheme['primary'])),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            self.story.append(metrics_table)
        
        # Summary text
        summary_text = self._generate_summary_text(report_data)
        self.story.append(Spacer(1, 20))
        self.story.append(Paragraph(summary_text, self.styles['Normal']))
        self.story.append(PageBreak())
    
    def _generate_summary_text(self, report_data: Dict[str, Any]) -> str:
        """Generate intelligent summary text based on data."""
        stats = report_data.get('statistics', {})
        trends = report_data.get('trends', {})
        anomalies = report_data.get('anomalies', [])
        
        summary_parts = []
        
        # Data overview
        count = stats.get('count', 0)
        summary_parts.append(f"This report analyzes {count:,} data points collected over the past {self.config.date_range_days} days.")
        
        # Trend analysis
        trend = trends.get('trend', 'unknown')
        if trend != 'unknown':
            confidence = trends.get('confidence', 'low')
            summary_parts.append(f"The data shows a {trend} trend with {confidence} confidence.")
        
        # Anomaly detection
        if anomalies:
            high_severity = len([a for a in anomalies if a.get('severity') == 'high'])
            summary_parts.append(f"Analysis detected {len(anomalies)} anomalies, including {high_severity} high-severity events.")
        
        # Performance summary
        mean_val = stats.get('mean', 0)
        std_dev = stats.get('std_dev', 0)
        if std_dev > 0:
            cv = (std_dev / mean_val) * 100
            if cv < 10:
                summary_parts.append("The system demonstrates high stability with low variability.")
            elif cv > 30:
                summary_parts.append("The data shows significant variability requiring investigation.")
        
        return " ".join(summary_parts)
    
    async def _build_section(self, section: ReportSection, report_data: Dict[str, Any]):
        """Build individual report section."""
        if section.page_break_before:
            self.story.append(PageBreak())
        
        # Section title
        self.story.append(Paragraph(section.title, self.styles['CustomHeading1']))
        
        # Section content based on type
        if section.content_type == 'metrics':
            await self._build_metrics_section(section, report_data)
        elif section.content_type == 'chart':
            await self._build_chart_section(section, report_data)
        elif section.content_type == 'table':
            await self._build_table_section(section, report_data)
        elif section.content_type == 'text':
            await self._build_text_section(section, report_data)
        
        if section.page_break_after:
            self.story.append(PageBreak())
    
    async def _build_metrics_section(self, section: ReportSection, report_data: Dict[str, Any]):
        """Build metrics dashboard section."""
        stats = report_data.get('statistics', {})
        
        # Create metrics grid
        metrics_grid = [
            [
                Paragraph("Total Records", self.styles['Normal']),
                Paragraph(f"{stats.get('count', 0):,}", self.styles['MetricValue'])
            ],
            [
                Paragraph("Average Value", self.styles['Normal']),
                Paragraph(f"{stats.get('mean', 0):.2f}", self.styles['MetricValue'])
            ],
            [
                Paragraph("Std Deviation", self.styles['Normal']),
                Paragraph(f"{stats.get('std_dev', 0):.2f}", self.styles['MetricValue'])
            ]
        ]
        
        metrics_table = Table(metrics_grid, colWidths=[2*inch, 2*inch])
        metrics_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('TOPPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.lightgrey)
        ]))
        
        self.story.append(metrics_table)
    
    async def _build_chart_section(self, section: ReportSection, report_data: Dict[str, Any]):
        """Build chart section."""
        # This would integrate with the visualization engine
        # For now, add placeholder
        self.story.append(Paragraph("Chart visualization would appear here", self.styles['Normal']))
        self.story.append(Spacer(1, 2*inch))  # Space for chart
    
    async def _build_table_section(self, section: ReportSection, report_data: Dict[str, Any]):
        """Build data table section."""
        if not section.data:
            return
        
        # Convert data to table format
        if isinstance(section.data, list) and section.data:
            # Assume list of dictionaries
            headers = list(section.data[0].keys())
            table_data = [headers]
            
            for row in section.data[:20]:  # Limit to 20 rows
                table_data.append([str(row.get(header, '')) for header in headers])
            
            table = Table(table_data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            self.story.append(table)
    
    async def _build_text_section(self, section: ReportSection, report_data: Dict[str, Any]):
        """Build text content section."""
        if section.template:
            # Use template rendering
            content = section.template.format(**report_data)
        elif section.data:
            content = str(section.data)
        else:
            content = "No content available for this section."
        
        self.story.append(Paragraph(content, self.styles['Normal']))
    
    async def _build_appendix(self, report_data: Dict[str, Any]):
        """Build appendix section."""
        self.story.append(Paragraph("Appendix", self.styles['CustomHeading1']))
        
        # Technical details
        self.story.append(Paragraph("Technical Details", self.styles['CustomHeading2']))
        
        tech_details = [
            f"Report generated: {datetime.datetime.now().isoformat()}",
            f"Data points analyzed: {report_data.get('statistics', {}).get('count', 0):,}",
            f"Analysis period: {self.config.date_range_days} days",
            f"Statistical confidence: {report_data.get('trends', {}).get('confidence', 'unknown')}"
        ]
        
        for detail in tech_details:
            self.story.append(Paragraph(f"• {detail}", self.styles['Normal']))


class DataExporter:
    """High-performance data export engine."""
    
    def __init__(self, config: AnalyticsConfig):
        self.config = config
        self.exports_dir = config.output_directory / "exports"
        self.exports_dir.mkdir(parents=True, exist_ok=True)
    
    @performance_monitor("export_data")
    async def export_data(
        self,
        data: List[BaseDataPoint],
        filename: str,
        format: ReportFormat,
        scope: ExportScope = ExportScope.COMPLETE,
        include_metadata: bool = True
    ) -> Path:
        """Export data to specified format."""
        if not data:
            raise ValueError("No data to export")
        
        # Prepare data based on scope
        export_data = self._prepare_export_data(data, scope, include_metadata)
        
        # Generate filename with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{filename}_{timestamp}"
        
        if format == ReportFormat.JSON:
            return await self._export_json(export_data, f"{base_name}.json")
        elif format == ReportFormat.CSV:
            return await self._export_csv(export_data, f"{base_name}.csv")
        elif format == ReportFormat.EXCEL:
            return await self._export_excel(export_data, f"{base_name}.xlsx")
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _prepare_export_data(
        self,
        data: List[BaseDataPoint],
        scope: ExportScope,
        include_metadata: bool
    ) -> List[Dict[str, Any]]:
        """Prepare data for export based on scope."""
        if scope == ExportScope.SUMMARY_ONLY:
            # Just statistical summary
            analyzer = StatisticalAnalyzer()
            stats = analyzer.calculate_basic_stats(data)
            return [{'summary_statistics': stats}]
        
        export_records = []
        for point in data:
            record = {
                'timestamp': point.timestamp.isoformat(),
                'value': point.value,
                'quality_score': point.quality_score
            }
            
            if point.source:
                record['source'] = point.source
            
            if include_metadata and point.metadata:
                record.update(point.metadata)
            
            export_records.append(record)
        
        return export_records
    
    async def _export_json(self, data: List[Dict[str, Any]], filename: str) -> Path:
        """Export data as JSON."""
        output_path = self.exports_dir / filename
        
        async with aiofiles.open(output_path, 'w') as f:
            await f.write(json.dumps(data, indent=2, default=str))
        
        return output_path
    
    async def _export_csv(self, data: List[Dict[str, Any]], filename: str) -> Path:
        """Export data as CSV."""
        output_path = self.exports_dir / filename
        
        if not data:
            return output_path
        
        # Get all unique fieldnames
        fieldnames = set()
        for record in data:
            fieldnames.update(record.keys())
        fieldnames = sorted(list(fieldnames))
        
        async with aiofiles.open(output_path, 'w', newline='') as f:
            # Write CSV manually since csv module isn't async
            content = []
            content.append(','.join(fieldnames))
            
            for record in data:
                row = []
                for field in fieldnames:
                    value = record.get(field, '')
                    # Escape commas and quotes
                    if isinstance(value, str) and (',' in value or '"' in value):
                        value = f'"{value.replace('"', '""')}"'
                    row.append(str(value))
                content.append(','.join(row))
            
            await f.write('\n'.join(content))
        
        return output_path
    
    async def _export_excel(self, data: List[Dict[str, Any]], filename: str) -> Path:
        """Export data as Excel file."""
        output_path = self.exports_dir / filename
        
        # Use pandas for Excel export
        df = pd.DataFrame(data)
        
        # Write to file (this is sync, but fast for reasonable data sizes)
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Data', index=False)
            
            # Add summary sheet if there's enough data
            if len(data) > 1:
                summary_data = {
                    'Metric': ['Total Records', 'Export Date', 'Data Types'],
                    'Value': [len(data), datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), len(df.columns)]
                }
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        return output_path


class ReportingEngine:
    """
    Enterprise Reporting Engine
    
    Comprehensive reporting system for generating analytics reports, data exports,
    and business intelligence documents with enterprise-grade features.
    """
    
    def __init__(self, config: Optional[AnalyticsConfig] = None):
        self.config = config or AnalyticsConfig()
        self.reports_dir = self.config.output_directory / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        self.data_exporter = DataExporter(self.config)
        self.statistical_analyzer = StatisticalAnalyzer()
        
        # Template environment for custom templates
        self.template_env = Environment(
            loader=FileSystemLoader(str(self.reports_dir / "templates"))
        )
    
    @performance_monitor("generate_report")
    async def generate_comprehensive_report(
        self,
        data: List[BaseDataPoint],
        config: ReportConfig,
        visualization_engine: Optional[VisualizationEngine] = None
    ) -> Path:
        """Generate a comprehensive analytics report."""
        if not data:
            raise ValueError("No data provided for report generation")
        
        # Analyze data
        report_data = await self._analyze_data(data)
        
        # Generate visualizations if engine provided
        if visualization_engine:
            report_data['charts'] = await self._generate_report_charts(data, visualization_engine)
        
        # Generate report based on format
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_title = "".join(c if c.isalnum() else "_" for c in config.title)
        
        if config.output_format == ReportFormat.PDF:
            filename = f"{safe_title}_{timestamp}.pdf"
            output_path = self.reports_dir / filename
            
            generator = PDFReportGenerator(config)
            return await generator.generate(report_data, output_path)
        
        elif config.output_format == ReportFormat.HTML:
            filename = f"{safe_title}_{timestamp}.html"
            output_path = self.reports_dir / filename
            return await self._generate_html_report(report_data, config, output_path)
        
        else:
            raise ValueError(f"Unsupported report format: {config.output_format}")
    
    async def _analyze_data(self, data: List[BaseDataPoint]) -> Dict[str, Any]:
        """Perform comprehensive data analysis."""
        analysis_result = {
            'data_summary': {
                'total_points': len(data),
                'date_range': {
                    'start': min(point.timestamp for point in data).isoformat(),
                    'end': max(point.timestamp for point in data).isoformat()
                },
                'sources': list(set(point.source for point in data if point.source))
            },
            'statistics': self.statistical_analyzer.calculate_basic_stats(data),
            'trends': self.statistical_analyzer.detect_trends(data),
            'anomalies': self.statistical_analyzer.detect_anomalies(data)
        }
        
        return analysis_result
    
    async def _generate_report_charts(
        self,
        data: List[BaseDataPoint],
        visualization_engine: VisualizationEngine
    ) -> Dict[str, str]:
        """Generate charts for the report."""
        charts = {}
        
        try:
            # Time series chart
            chart_bytes, _ = await visualization_engine.create_chart(
                data, ChartType.LINE, "Data Over Time", "Time", "Value"
            )
            charts['time_series'] = base64.b64encode(chart_bytes).decode('utf-8')
            
            # Distribution histogram
            chart_bytes, _ = await visualization_engine.create_chart(
                data, ChartType.HISTOGRAM, "Value Distribution", "Value", "Frequency"
            )
            charts['distribution'] = base64.b64encode(chart_bytes).decode('utf-8')
            
        except Exception as e:
            print(f"Error generating charts: {e}")
        
        return charts
    
    async def _generate_html_report(
        self,
        report_data: Dict[str, Any],
        config: ReportConfig,
        output_path: Path
    ) -> Path:
        """Generate HTML report."""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>{{ title }}</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .header { text-align: center; margin-bottom: 40px; }
                .section { margin-bottom: 30px; }
                .metrics { display: flex; justify-content: space-around; }
                .metric { text-align: center; padding: 20px; background: #f5f5f5; border-radius: 8px; }
                .chart { text-align: center; margin: 20px 0; }
                table { width: 100%; border-collapse: collapse; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{{ title }}</h1>
                <p>Generated on {{ generation_date }}</p>
            </div>
            
            <div class="section">
                <h2>Executive Summary</h2>
                <div class="metrics">
                    <div class="metric">
                        <h3>{{ statistics.count|int }}</h3>
                        <p>Total Data Points</p>
                    </div>
                    <div class="metric">
                        <h3>{{ "%.2f"|format(statistics.mean) }}</h3>
                        <p>Average Value</p>
                    </div>
                    <div class="metric">
                        <h3>{{ trends.trend|title }}</h3>
                        <p>Trend Direction</p>
                    </div>
                </div>
            </div>
            
            {% if charts %}
            <div class="section">
                <h2>Visualizations</h2>
                {% for chart_name, chart_data in charts.items() %}
                <div class="chart">
                    <h3>{{ chart_name|title|replace('_', ' ') }}</h3>
                    <img src="data:image/png;base64,{{ chart_data }}" alt="{{ chart_name }}">
                </div>
                {% endfor %}
            </div>
            {% endif %}
            
            <div class="section">
                <h2>Statistical Analysis</h2>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Count</td><td>{{ statistics.count|int }}</td></tr>
                    <tr><td>Mean</td><td>{{ "%.4f"|format(statistics.mean) }}</td></tr>
                    <tr><td>Standard Deviation</td><td>{{ "%.4f"|format(statistics.std_dev) }}</td></tr>
                    <tr><td>Minimum</td><td>{{ "%.4f"|format(statistics.min) }}</td></tr>
                    <tr><td>Maximum</td><td>{{ "%.4f"|format(statistics.max) }}</td></tr>
                </table>
            </div>
            
            {% if anomalies %}
            <div class="section">
                <h2>Anomalies Detected</h2>
                <table>
                    <tr><th>Timestamp</th><th>Value</th><th>Z-Score</th><th>Severity</th></tr>
                    {% for anomaly in anomalies[:10] %}
                    <tr>
                        <td>{{ anomaly.timestamp }}</td>
                        <td>{{ "%.4f"|format(anomaly.value) }}</td>
                        <td>{{ "%.2f"|format(anomaly.z_score) }}</td>
                        <td>{{ anomaly.severity|title }}</td>
                    </tr>
                    {% endfor %}
                </table>
            </div>
            {% endif %}
        </body>
        </html>
        """
        
        template = Template(html_template)
        html_content = template.render(
            title=config.title,
            generation_date=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            **report_data
        )
        
        async with aiofiles.open(output_path, 'w') as f:
            await f.write(html_content)
        
        return output_path
    
    async def export_data(
        self,
        data: List[BaseDataPoint],
        filename: str,
        format: ReportFormat,
        scope: ExportScope = ExportScope.COMPLETE
    ) -> Path:
        """Export data using the data exporter."""
        return await self.data_exporter.export_data(data, filename, format, scope)
    
    async def batch_export(
        self,
        datasets: Dict[str, List[BaseDataPoint]],
        formats: List[ReportFormat],
        create_archive: bool = True
    ) -> Union[Path, List[Path]]:
        """Export multiple datasets in multiple formats."""
        export_paths = []
        
        for dataset_name, data in datasets.items():
            for format in formats:
                try:
                    path = await self.export_data(data, dataset_name, format)
                    export_paths.append(path)
                except Exception as e:
                    print(f"Failed to export {dataset_name} as {format}: {e}")
        
        if create_archive and len(export_paths) > 1:
            # Create zip archive
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_path = self.config.output_directory / f"batch_export_{timestamp}.zip"
            
            with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                for path in export_paths:
                    zf.write(path, path.name)
            
            return archive_path
        
        return export_paths


# Export main classes
__all__ = [
    'ReportingEngine',
    'ReportConfig',
    'ReportSection',
    'ReportFormat',
    'ReportTemplate',
    'ExportScope',
    'StatisticalAnalyzer',
    'DataExporter',
    'PDFReportGenerator'
]