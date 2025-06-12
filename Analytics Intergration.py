"""
Enterprise Analytics Platform - Complete Integration Examples

This module demonstrates real-world integration patterns, advanced usage scenarios,
and production deployment examples for the enterprise analytics platform.

Examples include:
- Multi-source data integration
- Real-time dashboard generation
- Automated reporting pipelines
- Performance monitoring and alerting
- Compliance and audit reporting
- Custom data processors and analyzers
"""

import asyncio
import datetime
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

# Import all analytics components
from enterprise_analytics.core import (
    AnalyticsEngine, AnalyticsConfig, BaseDataPoint, MetricDefinition, 
    MetricType, DataSourceProtocol, ProcessorProtocol, performance_monitor
)
from enterprise_analytics.visualization import (
    VisualizationEngine, ChartType, ChartStyle, ColorScheme
)
from enterprise_analytics.reporting import (
    ReportingEngine, ReportConfig, ReportSection, ReportTemplate, 
    ReportFormat, ExportScope
)
from enterprise_logger import Logger, LogLevel

# Configure enterprise logging
logger = Logger("AnalyticsIntegration", {
    "minLevel": LogLevel.INFO,
    "logFile": "./logs/analytics_integration.log",
    "format": "structured",
    "enableBatching": True
})


class DatabaseDataSource:
    """Example database data source implementation."""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.connected = False
        logger.info("Database data source initialized", {"connection": connection_string})
    
    async def connect(self):
        """Simulate database connection."""
        await asyncio.sleep(0.1)  # Simulate connection delay
        self.connected = True
        logger.info("Database connection established")
    
    async def get_metrics(
        self, 
        start_time: datetime.datetime, 
        end_time: datetime.datetime,
        metric_names: Optional[List[str]] = None
    ) -> List[BaseDataPoint]:
        """Retrieve metrics from database."""
        if not self.connected:
            await self.connect()
        
        logger.debug("Fetching metrics from database", {
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "metric_names": metric_names
        })
        
        # Simulate database query
        data_points = []
        current_time = start_time
        
        while current_time <= end_time:
            # Simulate different metric types
            for metric_name in (metric_names or ["cpu_usage", "memory_usage", "disk_io"]):
                if metric_name == "cpu_usage":
                    value = random.uniform(10, 90)
                elif metric_name == "memory_usage":
                    value = random.uniform(30, 85)
                elif metric_name == "disk_io":
                    value = random.uniform(0, 1000)
                else:
                    value = random.uniform(0, 100)
                
                point = BaseDataPoint(
                    timestamp=current_time,
                    value=value,
                    source="production_database",
                    metadata={
                        "metric_type": metric_name,
                        "server": f"srv-{random.randint(1, 10):03d}",
                        "datacenter": random.choice(["us-east-1", "us-west-2", "eu-west-1"])
                    }
                )
                data_points.append(point)
            
            current_time += datetime.timedelta(minutes=5)
        
        logger.info("Metrics retrieved from database", {"count": len(data_points)})
        return data_points
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get database health status."""
        return {
            "status": "healthy" if self.connected else "disconnected",
            "connection_pool_size": 20,
            "active_connections": random.randint(5, 15),
            "query_latency_ms": random.uniform(10, 50),
            "last_query_time": datetime.datetime.now().isoformat()
        }


class APIDataSource:
    """Example API data source implementation."""
    
    def __init__(self, api_endpoint: str, api_key: str):
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.rate_limit_remaining = 1000
        logger.info("API data source initialized", {"endpoint": api_endpoint})
    
    async def get_metrics(
        self, 
        start_time: datetime.datetime, 
        end_time: datetime.datetime,
        metric_names: Optional[List[str]] = None
    ) -> List[BaseDataPoint]:
        """Retrieve metrics from external API."""
        logger.debug("Fetching metrics from API", {
            "endpoint": self.api_endpoint,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat()
        })
        
        # Simulate API rate limiting
        if self.rate_limit_remaining <= 0:
            logger.warn("API rate limit exceeded", {"endpoint": self.api_endpoint})
            return []
        
        self.rate_limit_remaining -= 1
        
        # Simulate API response delay
        await asyncio.sleep(0.2)
        
        # Generate sample data
        data_points = []
        duration_hours = (end_time - start_time).total_seconds() / 3600
        num_points = int(duration_hours * 4)  # Every 15 minutes
        
        for i in range(num_points):
            timestamp = start_time + datetime.timedelta(minutes=i * 15)
            
            # Simulate business metrics
            revenue = random.uniform(1000, 5000)
            user_count = random.randint(100, 2000)
            conversion_rate = random.uniform(0.02, 0.08)
            
            for metric_name, value in [
                ("revenue", revenue),
                ("active_users", user_count),
                ("conversion_rate", conversion_rate)
            ]:
                if not metric_names or metric_name in metric_names:
                    point = BaseDataPoint(
                        timestamp=timestamp,
                        value=value,
                        source="external_api",
                        metadata={
                            "metric_type": metric_name,
                            "campaign": f"campaign_{random.randint(1, 5)}",
                            "region": random.choice(["north", "south", "east", "west"])
                        }
                    )
                    data_points.append(point)
        
        logger.info("Metrics retrieved from API", {
            "count": len(data_points),
            "rate_limit_remaining": self.rate_limit_remaining
        })
        return data_points
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get API health status."""
        return {
            "status": "healthy",
            "rate_limit_remaining": self.rate_limit_remaining,
            "response_time_ms": random.uniform(100, 300),
            "api_version": "v2.1",
            "last_successful_call": datetime.datetime.now().isoformat()
        }


class AnomalyDetectionProcessor:
    """Custom processor for anomaly detection."""
    
    def __init__(self, sensitivity: float = 2.0):
        self.sensitivity = sensitivity
        self.baseline_cache = {}
        logger.info("Anomaly detection processor initialized", {"sensitivity": sensitivity})
    
    async def process(self, data: List[BaseDataPoint]) -> List[BaseDataPoint]:
        """Process data to detect and flag anomalies."""
        if len(data) < 10:
            return data  # Not enough data for anomaly detection
        
        # Group data by metric type
        metric_groups = {}
        for point in data:
            metric_type = point.metadata.get("metric_type", "unknown")
            if metric_type not in metric_groups:
                metric_groups[metric_type] = []
            metric_groups[metric_type].append(point)
        
        processed_data = []
        anomaly_count = 0
        
        for metric_type, points in metric_groups.items():
            values = [p.value for p in points]
            mean_val = np.mean(values)
            std_val = np.std(values)
            
            # Update baseline cache
            self.baseline_cache[metric_type] = {
                "mean": mean_val,
                "std": std_val,
                "last_updated": datetime.datetime.now()
            }
            
            # Flag anomalies
            for point in points:
                z_score = abs((point.value - mean_val) / std_val) if std_val > 0 else 0
                
                # Create new point with anomaly information
                new_metadata = dict(point.metadata)
                new_metadata.update({
                    "z_score": z_score,
                    "is_anomaly": z_score > self.sensitivity,
                    "anomaly_severity": "high" if z_score > 3.0 else "medium" if z_score > 2.0 else "low"
                })
                
                if z_score > self.sensitivity:
                    anomaly_count += 1
                
                new_point = BaseDataPoint(
                    timestamp=point.timestamp,
                    value=point.value,
                    source=point.source,
                    metadata=new_metadata,
                    quality_score=max(0.0, 1.0 - (z_score / 10.0))
                )
                processed_data.append(new_point)
        
        logger.info("Anomaly detection completed", {
            "total_points": len(data),
            "anomalies_detected": anomaly_count,
            "anomaly_rate": anomaly_count / len(data) if data else 0
        })
        
        return processed_data


class ComplianceReportGenerator:
    """Specialized compliance report generator."""
    
    def __init__(self, reporting_engine: ReportingEngine):
        self.reporting_engine = reporting_engine
        logger.info("Compliance report generator initialized")
    
    async def generate_sox_compliance_report(
        self,
        data: List[BaseDataPoint],
        period_start: datetime.datetime,
        period_end: datetime.datetime
    ) -> Path:
        """Generate SOX compliance report."""
        logger.info("Generating SOX compliance report", {
            "period_start": period_start.isoformat(),
            "period_end": period_end.isoformat(),
            "data_points": len(data)
        })
        
        # Define compliance-specific sections
        sections = [
            ReportSection(
                title="Executive Summary",
                content_type="text",
                template="This report covers the period from {start} to {end} for SOX compliance monitoring.",
                include_in_toc=True
            ),
            ReportSection(
                title="Data Integrity Analysis",
                content_type="metrics",
                include_in_toc=True
            ),
            ReportSection(
                title="Control Effectiveness",
                content_type="table",
                include_in_toc=True
            ),
            ReportSection(
                title="Exception Analysis",
                content_type="chart",
                include_in_toc=True
            ),
            ReportSection(
                title="Audit Trail",
                content_type="table",
                include_in_toc=True
            )
        ]
        
        # Configure compliance report
        report_config = ReportConfig(
            title="SOX Compliance Report",
            subtitle=f"Period: {period_start.strftime('%Y-%m-%d')} to {period_end.strftime('%Y-%m-%d')}",
            author="Compliance Analytics System",
            organization="Enterprise Corporation",
            template=ReportTemplate.COMPLIANCE_AUDIT,
            sections=sections,
            include_toc=True,
            include_executive_summary=True,
            statistical_analysis=True,
            output_format=ReportFormat.PDF,
            watermark="CONFIDENTIAL"
        )
        
        # Generate the report
        report_path = await self.reporting_engine.generate_comprehensive_report(
            data, report_config
        )
        
        logger.info("SOX compliance report generated", {"path": str(report_path)})
        return report_path


class RealTimeDashboard:
    """Real-time analytics dashboard manager."""
    
    def __init__(self, analytics_engine: AnalyticsEngine, viz_engine: VisualizationEngine):
        self.analytics_engine = analytics_engine
        self.viz_engine = viz_engine
        self.dashboard_cache = {}
        self.update_interval = 30  # seconds
        logger.info("Real-time dashboard initialized")
    
    async def generate_executive_dashboard(self) -> Dict[str, Any]:
        """Generate executive dashboard with key metrics."""
        logger.debug("Generating executive dashboard")
        
        # Collect recent data from all sources
        end_time = datetime.datetime.now()
        start_time = end_time - datetime.timedelta(hours=24)
        
        dashboard_data = {
            "timestamp": end_time.isoformat(),
            "charts": {},
            "metrics": {},
            "alerts": []
        }
        
        try:
            # Get system metrics
            system_data = await self.analytics_engine.collect_metrics(
                "production_database", start_time, end_time, ["cpu_usage", "memory_usage"]
            )
            
            # Get business metrics
            business_data = await self.analytics_engine.collect_metrics(
                "external_api", start_time, end_time, ["revenue", "active_users"]
            )
            
            # Generate charts
            if system_data:
                chart_bytes, _ = await self.viz_engine.create_chart(
                    system_data,
                    ChartType.LINE,
                    "System Performance - 24h",
                    "Time",
                    "Usage %"
                )
                dashboard_data["charts"]["system_performance"] = chart_bytes
            
            if business_data:
                chart_bytes, _ = await self.viz_engine.create_chart(
                    business_data,
                    ChartType.AREA,
                    "Business Metrics - 24h",
                    "Time",
                    "Value"
                )
                dashboard_data["charts"]["business_metrics"] = chart_bytes
            
            # Calculate key metrics
            all_data = system_data + business_data
            if all_data:
                anomalies = [p for p in all_data if p.metadata.get("is_anomaly", False)]
                
                dashboard_data["metrics"] = {
                    "total_data_points": len(all_data),
                    "anomalies_detected": len(anomalies),
                    "system_health": "healthy" if len(anomalies) < 10 else "warning",
                    "last_updated": end_time.isoformat()
                }
                
                # Generate alerts for high-severity anomalies
                high_severity_anomalies = [
                    p for p in anomalies 
                    if p.metadata.get("anomaly_severity") == "high"
                ]
                
                for anomaly in high_severity_anomalies[:5]:  # Limit to 5 alerts
                    dashboard_data["alerts"].append({
                        "timestamp": anomaly.timestamp.isoformat(),
                        "metric": anomaly.metadata.get("metric_type", "unknown"),
                        "value": anomaly.value,
                        "z_score": anomaly.metadata.get("z_score", 0),
                        "severity": "high"
                    })
        
        except Exception as e:
            logger.error("Error generating executive dashboard", error=e)
            dashboard_data["error"] = str(e)
        
        logger.info("Executive dashboard generated", {
            "charts_count": len(dashboard_data["charts"]),
            "alerts_count": len(dashboard_data["alerts"])
        })
        
        return dashboard_data
    
    async def start_real_time_updates(self, callback: callable = None):
        """Start real-time dashboard updates."""
        logger.info("Starting real-time dashboard updates", {
            "update_interval": self.update_interval
        })
        
        while True:
            try:
                dashboard = await self.generate_executive_dashboard()
                self.dashboard_cache = dashboard
                
                if callback:
                    await callback(dashboard)
                
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                logger.error("Error in real-time dashboard update", error=e)
                await asyncio.sleep(self.update_interval)


class AnalyticsPipeline:
    """Complete analytics pipeline orchestrator."""
    
    def __init__(self):
        # Initialize configuration
        self.config = AnalyticsConfig(
            output_directory=Path("./enterprise_analytics_output"),
            data_retention_days=90,
            worker_pool_size=8,
            cache_ttl_seconds=1800,
            enable_streaming=True,
            enable_performance_monitoring=True
        )
        
        # Initialize core engines
        self.analytics_engine = AnalyticsEngine(self.config)
        self.viz_engine = VisualizationEngine(self.config)
        self.reporting_engine = ReportingEngine(self.config)
        
        # Initialize specialized components
        self.compliance_generator = ComplianceReportGenerator(self.reporting_engine)
        self.dashboard = RealTimeDashboard(self.analytics_engine, self.viz_engine)
        
        logger.info("Analytics pipeline initialized")
    
    async def setup_data_sources(self):
        """Set up and register all data sources."""
        logger.info("Setting up data sources")
        
        # Register database source
        db_source = DatabaseDataSource("postgresql://analytics:password@db:5432/metrics")
        await self.analytics_engine.register_data_source("production_database", db_source)
        
        # Register API source
        api_source = APIDataSource("https://api.example.com/metrics", "api_key_123")
        await self.analytics_engine.register_data_source("external_api", api_source)
        
        # Register metrics definitions
        metrics = [
            MetricDefinition(
                name="cpu_usage",
                type=MetricType.GAUGE,
                unit="percent",
                description="CPU utilization percentage",
                min_value=0.0,
                max_value=100.0,
                retention_days=90
            ),
            MetricDefinition(
                name="memory_usage",
                type=MetricType.GAUGE,
                unit="percent",
                description="Memory utilization percentage",
                min_value=0.0,
                max_value=100.0,
                retention_days=90
            ),
            MetricDefinition(
                name="revenue",
                type=MetricType.COUNTER,
                unit="dollars",
                description="Revenue in USD",
                min_value=0.0,
                retention_days=365
            )
        ]
        
        for metric in metrics:
            await self.analytics_engine.register_metric(metric)
        
        # Add custom processors
        anomaly_processor = AnomalyDetectionProcessor(sensitivity=2.5)
        await self.analytics_engine.add_processor(anomaly_processor)
        
        logger.info("Data sources setup completed")
    
    async def run_daily_analytics_job(self):
        """Run comprehensive daily analytics job."""
        logger.info("Starting daily analytics job")
        start_time = time.time()
        
        try:
            # Define time range
            end_time = datetime.datetime.now()
            start_time_dt = end_time - datetime.timedelta(days=1)
            
            # Collect all metrics
            logger.info("Collecting metrics from all sources")
            
            system_metrics = await self.analytics_engine.collect_metrics(
                "production_database", start_time_dt, end_time
            )
            
            business_metrics = await self.analytics_engine.collect_metrics(
                "external_api", start_time_dt, end_time
            )
            
            all_metrics = system_metrics + business_metrics
            logger.info("Metrics collection completed", {"total_points": len(all_metrics)})
            
            # Generate reports
            logger.info("Generating daily reports")
            
            # Technical report
            tech_report_config = ReportConfig(
                title="Daily Technical Analytics Report",
                subtitle=f"System Performance - {end_time.strftime('%Y-%m-%d')}",
                template=ReportTemplate.TECHNICAL_ANALYSIS,
                output_format=ReportFormat.PDF
            )
            
            tech_report_path = await self.reporting_engine.generate_comprehensive_report(
                system_metrics, tech_report_config, self.viz_engine
            )
            
            # Business report
            business_report_config = ReportConfig(
                title="Daily Business Analytics Report",
                subtitle=f"Business Metrics - {end_time.strftime('%Y-%m-%d')}",
                template=ReportTemplate.EXECUTIVE_SUMMARY,
                output_format=ReportFormat.HTML
            )
            
            business_report_path = await self.reporting_engine.generate_comprehensive_report(
                business_metrics, business_report_config, self.viz_engine
            )
            
            # Compliance report (weekly)
            if end_time.weekday() == 6:  # Sunday
                compliance_report_path = await self.compliance_generator.generate_sox_compliance_report(
                    all_metrics, start_time_dt, end_time
                )
                logger.info("Weekly compliance report generated", {"path": str(compliance_report_path)})
            
            # Export data for backup
            logger.info("Exporting data for backup")
            
            export_path = await self.reporting_engine.export_data(
                all_metrics, "daily_backup", ReportFormat.JSON
            )
            
            # Performance metrics
            performance_metrics = await self.analytics_engine.get_performance_metrics()
            
            duration = time.time() - start_time
            logger.info("Daily analytics job completed", {
                "duration_seconds": duration,
                "tech_report": str(tech_report_path),
                "business_report": str(business_report_path),
                "export_path": str(export_path),
                "performance_metrics": performance_metrics
            })
            
            return {
                "status": "success",
                "duration": duration,
                "reports_generated": 2,
                "data_points_processed": len(all_metrics),
                "performance": performance_metrics
            }
            
        except Exception as e:
            logger.error("Daily analytics job failed", error=e)
            return {
                "status": "failed",
                "error": str(e),
                "duration": time.time() - start_time
            }
    
    async def run_real_time_monitoring(self):
        """Start real-time monitoring and alerting."""
        logger.info("Starting real-time monitoring")
        
        async def alert_callback(dashboard_data):
            """Handle real-time alerts."""
            alerts = dashboard_data.get("alerts", [])
            if alerts:
                logger.warn("High-severity alerts detected", {
                    "alert_count": len(alerts),
                    "alerts": alerts
                })
                
                # In production, you would send these to your alerting system
                # (PagerDuty, Slack, email, etc.)
        
        # Start real-time dashboard updates
        await self.dashboard.start_real_time_updates(callback=alert_callback)
    
    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive system health check."""
        logger.debug("Performing health check")
        
        health_status = {
            "timestamp": datetime.datetime.now().isoformat(),
            "overall_status": "healthy",
            "components": {}
        }
        
        try:
            # Analytics engine health
            analytics_health = await self.analytics_engine.health_check()
            health_status["components"]["analytics_engine"] = analytics_health
            
            # Visualization engine health
            viz_stats = self.viz_engine.get_cache_stats()
            health_status["components"]["visualization_engine"] = {
                "status": "healthy",
                "cache_stats": viz_stats
            }
            
            # Check if any component is unhealthy
            for component, status in health_status["components"].items():
                if status.get("status") != "healthy":
                    health_status["overall_status"] = "degraded"
                    break
        
        except Exception as e:
            logger.error("Health check failed", error=e)
            health_status["overall_status"] = "unhealthy"
            health_status["error"] = str(e)
        
        return health_status
    
    async def cleanup(self):
        """Clean up all resources."""
        logger.info("Cleaning up analytics pipeline")
        
        try:
            await self.analytics_engine.close()
            await self.viz_engine.clear_cache()
            logger.info("Analytics pipeline cleanup completed")
        except Exception as e:
            logger.error("Error during cleanup", error=e)


# Example usage and integration patterns
async def main():
    """Main example demonstrating complete analytics pipeline."""
    logger.info("Starting enterprise analytics pipeline example")
    
    # Initialize pipeline
    pipeline = AnalyticsPipeline()
    
    try:
        # Set up data sources and processors
        await pipeline.setup_data_sources()
        
        # Run health check
        health = await pipeline.health_check()
        logger.info("Initial health check", {"status": health["overall_status"]})
        
        # Run daily analytics job
        job_result = await pipeline.run_daily_analytics_job()
        logger.info("Daily analytics job result", job_result)
        
        # Generate executive dashboard
        dashboard = await pipeline.dashboard.generate_executive_dashboard()
        logger.info("Executive dashboard generated", {
            "charts": len(dashboard["charts"]),
            "alerts": len(dashboard["alerts"])
        })
        
        # Example: Run real-time monitoring for 5 minutes
        logger.info("Starting 5-minute real-time monitoring demo")
        
        monitoring_task = asyncio.create_task(pipeline.run_real_time_monitoring())
        
        # Let it run for 5 minutes
        await asyncio.sleep(300)
        
        # Cancel monitoring
        monitoring_task.cancel()
        
        logger.info("Real-time monitoring demo completed")
        
    except Exception as e:
        logger.error("Pipeline execution failed", error=e)
        raise
    finally:
        # Clean up
        await pipeline.cleanup()
        logger.info("Enterprise analytics pipeline example completed")


if __name__ == "__main__":
    # Run the complete example
    asyncio.run(main())


# Additional utility functions for production deployment
class ProductionDeploymentHelper:
    """Helper class for production deployment configurations."""
    
    @staticmethod
    def get_kubernetes_config() -> Dict[str, Any]:
        """Get Kubernetes deployment configuration."""
        return {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": "enterprise-analytics",
                "labels": {"app": "enterprise-analytics"}
            },
            "spec": {
                "replicas": int(os.getenv("ANALYTICS_REPLICAS", "3")),
                "selector": {
                    "matchLabels": {"app": "enterprise-analytics"}
                },
                "template": {
                    "metadata": {
                        "labels": {"app": "enterprise-analytics"}
                    },
                    "spec": {
                        "containers": [{
                            "name": "analytics",
                            "image": os.getenv("ANALYTICS_IMAGE", "enterprise-analytics:latest"),
                            "ports": [{"containerPort": 8080}],
                            "env": [
                                {"name": "ANALYTICS_OUTPUT_DIR", "value": "/data/analytics"},
                                {"name": "REDIS_URL", "value": "redis://redis-service:6379"},
                                {"name": "DATABASE_URL", "value": os.getenv("DATABASE_URL")},
                                {"name": "LOG_LEVEL", "value": "INFO"}
                            ],
                            "resources": {
                                "requests": {"memory": "2Gi", "cpu": "1000m"},
                                "limits": {"memory": "4Gi", "cpu": "2000m"}
                            },
                            "volumeMounts": [{
                                "name": "analytics-storage",
                                "mountPath": "/data/analytics"
                            }]
                        }],
                        "volumes": [{
                            "name": "analytics-storage",
                            "persistentVolumeClaim": {
                                "claimName": "analytics-pvc"
                            }
                        }]
                    }
                }
            }
        }
    
    @staticmethod
    def get_docker_compose_config() -> Dict[str, Any]:
        """Get Docker Compose configuration."""
        return {
            "version": "3.8",
            "services": {
                "analytics": {
                    "build": ".",
                    "ports": ["8080:8080"],
                    "environment": {
                        "ANALYTICS_OUTPUT_DIR": "/data/analytics",
                        "DATABASE_URL": "postgresql://user:pass@postgres:5432/analytics",
                        "REDIS_URL": "redis://redis:6379",
                        "LOG_LEVEL": "INFO"
                    },
                    "volumes": ["./data:/data/analytics"],
                    "depends_on": ["postgres", "redis"],
                    "restart": "unless-stopped"
                },
                "postgres": {
                    "image": "postgres:13",
                    "environment": {
                        "POSTGRES_DB": "analytics",
                        "POSTGRES_USER": "user",
                        "POSTGRES_PASSWORD": "pass"
                    },
                    "volumes": ["postgres_data:/var/lib/postgresql/data"]
                },
                "redis": {
                    "image": "redis:6-alpine",
                    "volumes": ["redis_data:/data"]
                }
            },
            "volumes": {
                "postgres_data": {},
                "redis_data": {}
            }
        }


# Export all classes for easy importing
__all__ = [
    'DatabaseDataSource',
    'APIDataSource', 
    'AnomalyDetectionProcessor',
    'ComplianceReportGenerator',
    'RealTimeDashboard',
    'AnalyticsPipeline',
    'ProductionDeploymentHelper'
]