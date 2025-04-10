"""
Performance monitoring for RAG system.

This module implements the monitoring framework described in Specification.md
section 2.8, providing comprehensive performance tracking, alerting, and
visualization capabilities for the RAG engine.
"""
import time
import threading
import json
from typing import Dict, Any, List, Optional, Union, Callable
from pathlib import Path
from datetime import datetime, timedelta
from enum import Enum
from collections import deque
import statistics

from oarc_rag.utils.log import log
from oarc_rag.utils.decorators.singleton import singleton
from oarc_rag.utils.config.config import Config


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(Enum):
    """Types of metrics being monitored."""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    VECTOR_QUALITY = "vector_quality"
    RESOURCE_USAGE = "resource_usage"
    RETRIEVAL_QUALITY = "retrieval_quality"
    CONTEXT_RELEVANCE = "context_relevance"
    EMBEDDING_PERFORMANCE = "embedding_performance"
    CACHE_EFFICIENCY = "cache_efficiency"


@singleton
class RAGMonitor:
    """
    Performance monitoring system for the RAG engine.
    
    This class implements comprehensive monitoring as described in
    Specification.md section 2.8, tracking performance metrics,
    generating alerts, and supporting visualization.
    """
    
    def __init__(
        self,
        metrics_dir: Optional[Union[str, Path]] = None,
        history_size: int = 1000,
        alert_handlers: Optional[List[Callable]] = None,
        enable_periodic_reporting: bool = False,
        reporting_interval: int = 3600  # 1 hour
    ):
        """
        Initialize the RAG monitor.
        
        Args:
            metrics_dir: Directory for storing metrics
            history_size: Number of data points to keep in memory
            alert_handlers: Functions to call when alerts are generated
            enable_periodic_reporting: Whether to enable periodic reporting
            reporting_interval: Seconds between periodic reports
        """
        # Set metrics directory
        config = Config()
        if metrics_dir is None:
            metrics_dir = config.get('monitoring.metrics_dir', None)
            if metrics_dir is None:
                metrics_dir = Path.cwd() / "metrics"
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        
        # Metrics storage
        self.history_size = history_size
        self.metrics: Dict[str, deque] = {}
        self.aggregated_metrics: Dict[str, Dict[str, Any]] = {}
        self.custom_metrics: Dict[str, Dict[str, Any]] = {}
        self.alerts: List[Dict[str, Any]] = []
        self.alert_handlers = alert_handlers or []
        
        # Component metrics
        self.engine_metrics: Dict[str, Any] = {}
        self.database_metrics: Dict[str, Any] = {}
        self.embedding_metrics: Dict[str, Any] = {}
        self.vector_metrics: Dict[str, Any] = {}
        self.retrieval_metrics: Dict[str, Any] = {}
        
        # System resource monitoring
        self.resource_metrics: Dict[str, Any] = {
            "memory_usage": [],
            "cpu_usage": [],
            "disk_usage": [],
            "average_memory_usage": 0.0,
            "average_cpu_usage": 0.0,
            "peak_memory_usage": 0.0,
            "peak_cpu_usage": 0.0
        }
        
        # Thresholds for alerts based on config or defaults
        self.thresholds = {
            "latency_warning": config.get("monitoring.threshold.latency_warning", 2.0),
            "latency_critical": config.get("monitoring.threshold.latency_critical", 5.0),
            "memory_warning": config.get("monitoring.threshold.memory_warning", 80.0),
            "memory_critical": config.get("monitoring.threshold.memory_critical", 95.0),
            "cpu_warning": config.get("monitoring.threshold.cpu_warning", 80.0),
            "cpu_critical": config.get("monitoring.threshold.cpu_critical", 95.0),
            "similarity_warning": config.get("monitoring.threshold.similarity_warning", 0.3),
            "cache_miss_warning": config.get("monitoring.threshold.cache_miss_warning", 0.7)
        }
        
        # Initialize counter metrics
        self._initialize_metrics()
        
        # For implementing concepts from Big_Brain.md
        # - Track the sleep/awake phases
        self.mode_history: List[Dict[str, Any]] = []
        self.mode_transitions = 0
        self.current_mode = "awake"  # Default to awake mode
        
        # Set up periodic reporting if enabled
        self.enable_periodic_reporting = enable_periodic_reporting
        self.reporting_interval = reporting_interval
        self._reporting_thread = None
        
        if self.enable_periodic_reporting:
            self._start_periodic_reporting()
            
        log.info(f"RAG Monitor initialized with metrics directory: {self.metrics_dir}")
        
    def _initialize_metrics(self) -> None:
        """Initialize metrics storage."""
        # Performance metrics
        for metric_type in MetricType:
            self.metrics[metric_type.value] = deque(maxlen=self.history_size)
            self.aggregated_metrics[metric_type.value] = {
                "count": 0,
                "avg": 0.0,
                "min": float('inf'),
                "max": 0.0,
                "p90": 0.0,  # 90th percentile
                "p99": 0.0,  # 99th percentile
                "last_update": time.time(),
                "trend": "stable"  # stable, improving, degrading
            }
            
        # Initialize operational metrics
        self.operational_metrics = {
            "start_time": time.time(),
            "uptime": 0.0,
            "total_requests": 0,
            "requests_per_second": 0.0,
            "error_count": 0,
            "error_rate": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
            "cache_hit_ratio": 0.0
        }
        
    def log_metric(self, metric_type: Union[str, MetricType], value: float) -> None:
        """
        Log a performance metric.
        
        Args:
            metric_type: Type of metric (can be string or MetricType enum)
            value: Metric value
            
        Raises:
            ValueError: If metric type is not recognized
        """
        # Convert enum to string if needed
        if isinstance(metric_type, MetricType):
            metric_type = metric_type.value
            
        # Validate metric type
        if metric_type not in [m.value for m in MetricType]:
            raise ValueError(f"Unrecognized metric type: {metric_type}")
            
        # Record metric
        self.metrics[metric_type].append((time.time(), value))
        
        # Update aggregated metrics
        aggregated = self.aggregated_metrics[metric_type]
        aggregated["count"] += 1
        
        # Calculate running average
        prev_avg = aggregated["avg"]
        aggregated["avg"] = prev_avg + (value - prev_avg) / aggregated["count"]
        
        # Update min/max
        aggregated["min"] = min(aggregated["min"], value)
        aggregated["max"] = max(aggregated["max"], value)
        
        # Update percentiles if we have enough data
        metric_values = [v for _, v in self.metrics[metric_type]]
        if len(metric_values) >= 10:
            sorted_values = sorted(metric_values)
            aggregated["p90"] = sorted_values[int(0.9 * len(sorted_values))]
            aggregated["p99"] = sorted_values[int(0.99 * len(sorted_values))]
            
        # Determine trend
        if aggregated["count"] > 1:
            prev_trend = aggregated["trend"]
            if value < prev_avg:
                new_trend = "improving" if metric_type in ["latency"] else "degrading"
            elif value > prev_avg:
                new_trend = "degrading" if metric_type in ["latency"] else "improving"
            else:
                new_trend = "stable"
                
            # Only change trend if consistent over several data points
            if new_trend != prev_trend:
                trend_values = [v for _, v in list(self.metrics[metric_type])[-5:]]
                trend_avg = sum(trend_values) / len(trend_values)
                
                if ((trend_avg < prev_avg and metric_type in ["latency"]) or 
                    (trend_avg > prev_avg and metric_type not in ["latency"])):
                    aggregated["trend"] = "improving"
                elif ((trend_avg > prev_avg and metric_type in ["latency"]) or
                      (trend_avg < prev_avg and metric_type not in ["latency"])):
                    aggregated["trend"] = "degrading"
                else:
                    aggregated["trend"] = "stable"
        
        # Check for threshold alerts
        self._check_metric_thresholds(metric_type, value)
        
    def log_custom_metric(self, name: str, value: Any, category: str = "custom") -> None:
        """
        Log a custom metric not part of standard types.
        
        Args:
            name: Custom metric name
            value: Metric value
            category: Category for grouping metrics
        """
        if category not in self.custom_metrics:
            self.custom_metrics[category] = {}
            
        self.custom_metrics[category][name] = {
            "value": value,
            "timestamp": time.time()
        }
        
    def _check_metric_thresholds(self, metric_type: str, value: float) -> None:
        """
        Check if metric exceeds configured thresholds and generate alerts.
        
        Args:
            metric_type: Type of metric
            value: Current metric value
        """
        # Check latency thresholds
        if metric_type == "latency":
            if value > self.thresholds["latency_critical"]:
                self._create_alert(
                    f"Critical latency: {value:.2f}s exceeds threshold of {self.thresholds['latency_critical']}s",
                    AlertLevel.CRITICAL,
                    metric_type,
                    value
                )
            elif value > self.thresholds["latency_warning"]:
                self._create_alert(
                    f"High latency: {value:.2f}s exceeds threshold of {self.thresholds['latency_warning']}s",
                    AlertLevel.WARNING,
                    metric_type,
                    value
                )
                
        # Check retrieval quality threshold
        elif metric_type == "retrieval_quality" and value < self.thresholds["similarity_warning"]:
            self._create_alert(
                f"Low retrieval quality: {value:.2f} below threshold of {self.thresholds['similarity_warning']}",
                AlertLevel.WARNING,
                metric_type,
                value
            )
            
        # Check cache efficiency threshold
        elif metric_type == "cache_efficiency" and value < self.thresholds["cache_miss_warning"]:
            self._create_alert(
                f"Low cache efficiency: {value:.2f} below threshold of {self.thresholds['cache_miss_warning']}",
                AlertLevel.WARNING,
                metric_type,
                value
            )
            
    def _create_alert(
        self, 
        message: str, 
        level: AlertLevel, 
        metric_type: str, 
        value: Any
    ) -> None:
        """
        Create and record an alert.
        
        Args:
            message: Alert message
            level: Alert severity level
            metric_type: Type of metric triggering the alert
            value: Metric value
        """
        alert = {
            "message": message,
            "level": level.value,
            "metric_type": metric_type,
            "value": value,
            "timestamp": time.time()
        }
        
        # Record the alert
        self.alerts.append(alert)
        
        # Log the alert
        log_methods = {
            AlertLevel.INFO: log.info,
            AlertLevel.WARNING: log.warning,
            AlertLevel.ERROR: log.error,
            AlertLevel.CRITICAL: log.critical
        }
        log_methods[level](f"ALERT: {message}")
        
        # Call alert handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                log.error(f"Error in alert handler: {e}")
                
    def register_alert_handler(self, handler: Callable) -> None:
        """
        Register a function to be called when alerts are generated.
        
        Args:
            handler: Function to call with alert dictionary
        """
        self.alert_handlers.append(handler)
        
    def update_operational_metrics(
        self, 
        request_count: Optional[int] = None,
        error_count: Optional[int] = None,
        cache_hit: Optional[bool] = None
    ) -> None:
        """
        Update operational metrics for the RAG system.
        
        Args:
            request_count: Number of requests to add (None to just update uptime)
            error_count: Number of errors to add
            cache_hit: Whether a cache hit occurred
        """
        # Update uptime
        self.operational_metrics["uptime"] = time.time() - self.operational_metrics["start_time"]
        
        # Update request counters
        if request_count is not None:
            self.operational_metrics["total_requests"] += request_count
            self.operational_metrics["requests_per_second"] = (
                self.operational_metrics["total_requests"] / self.operational_metrics["uptime"]
                if self.operational_metrics["uptime"] > 0 else 0
            )
            
        # Update error metrics
        if error_count is not None:
            self.operational_metrics["error_count"] += error_count
            self.operational_metrics["error_rate"] = (
                self.operational_metrics["error_count"] / 
                max(1, self.operational_metrics["total_requests"])
            )
            
        # Update cache metrics
        if cache_hit is not None:
            if cache_hit:
                self.operational_metrics["cache_hits"] += 1
            else:
                self.operational_metrics["cache_misses"] += 1
                
            total_cache = (
                self.operational_metrics["cache_hits"] + 
                self.operational_metrics["cache_misses"]
            )
            
            self.operational_metrics["cache_hit_ratio"] = (
                self.operational_metrics["cache_hits"] / total_cache if total_cache > 0 else 0
            )
            
    def update_resource_metrics(
        self, 
        memory_percent: Optional[float] = None,
        cpu_percent: Optional[float] = None,
        disk_percent: Optional[float] = None
    ) -> None:
        """
        Update system resource metrics.
        
        Args:
            memory_percent: Memory usage as percentage
            cpu_percent: CPU usage as percentage
            disk_percent: Disk usage as percentage
        """
        timestamp = time.time()
        
        if memory_percent is not None:
            self.resource_metrics["memory_usage"].append((timestamp, memory_percent))
            self.resource_metrics["average_memory_usage"] = (
                statistics.mean([x[1] for x in self.resource_metrics["memory_usage"][-10:]])
            )
            self.resource_metrics["peak_memory_usage"] = max(
                self.resource_metrics["peak_memory_usage"],
                memory_percent
            )
            
            # Check thresholds for alerts
            if memory_percent > self.thresholds["memory_critical"]:
                self._create_alert(
                    f"Critical memory usage: {memory_percent:.1f}%",
                    AlertLevel.CRITICAL,
                    "memory_usage",
                    memory_percent
                )
            elif memory_percent > self.thresholds["memory_warning"]:
                self._create_alert(
                    f"High memory usage: {memory_percent:.1f}%",
                    AlertLevel.WARNING,
                    "memory_usage",
                    memory_percent
                )
                
        if cpu_percent is not None:
            self.resource_metrics["cpu_usage"].append((timestamp, cpu_percent))
            self.resource_metrics["average_cpu_usage"] = (
                statistics.mean([x[1] for x in self.resource_metrics["cpu_usage"][-10:]])
            )
            self.resource_metrics["peak_cpu_usage"] = max(
                self.resource_metrics["peak_cpu_usage"],
                cpu_percent
            )
            
            # Check thresholds for alerts
            if cpu_percent > self.thresholds["cpu_critical"]:
                self._create_alert(
                    f"Critical CPU usage: {cpu_percent:.1f}%",
                    AlertLevel.CRITICAL,
                    "cpu_usage",
                    cpu_percent
                )
            elif cpu_percent > self.thresholds["cpu_warning"]:
                self._create_alert(
                    f"High CPU usage: {cpu_percent:.1f}%",
                    AlertLevel.WARNING,
                    "cpu_usage",
                    cpu_percent
                )
                
        if disk_percent is not None:
            self.resource_metrics["disk_usage"].append((timestamp, disk_percent))
            
    def update_engine_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Update RAG engine metrics.
        
        Args:
            metrics: Dictionary of engine metrics
        """
        self.engine_metrics.update(metrics)
        
    def update_database_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Update vector database metrics.
        
        Args:
            metrics: Dictionary of database metrics
        """
        self.database_metrics.update(metrics)
        
    def update_embedding_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Update embedding generation metrics.
        
        Args:
            metrics: Dictionary of embedding metrics
        """
        self.embedding_metrics.update(metrics)
        
    def register_mode_transition(self, old_mode: str, new_mode: str) -> None:
        """
        Register a transition between operational modes (from Big_Brain.md).
        
        Args:
            old_mode: Previous operational mode
            new_mode: New operational mode
        """
        transition = {
            "timestamp": time.time(),
            "from_mode": old_mode,
            "to_mode": new_mode
        }
        
        self.mode_history.append(transition)
        self.mode_transitions += 1
        self.current_mode = new_mode
        
        # Log the transition
        log.info(f"Operational mode transition: {old_mode} → {new_mode}")
        
        # Track custom metric for mode transitions
        self.log_custom_metric("mode_transition", {
            "from": old_mode,
            "to": new_mode
        }, category="operational_modes")
        
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive metrics report.
        
        Returns:
            Dict containing all metrics
        """
        return {
            "aggregated_metrics": self.aggregated_metrics,
            "operational_metrics": self.operational_metrics,
            "resource_metrics": {k: v for k, v in self.resource_metrics.items() if not isinstance(v, list)},
            "engine_metrics": self.engine_metrics,
            "database_metrics": self.database_metrics,
            "embedding_metrics": self.embedding_metrics,
            "custom_metrics": self.custom_metrics,
            "alerts": self.alerts[-10:],  # Last 10 alerts
            "alert_count": len(self.alerts),
            "operational_mode": {
                "current_mode": self.current_mode,
                "mode_transitions": self.mode_transitions
            }
        }
        
    def get_time_series_metrics(self, metric_type: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get time series data for a specific metric.
        
        Args:
            metric_type: Type of metric
            limit: Maximum number of data points
            
        Returns:
            List of data points
        """
        if metric_type not in self.metrics:
            return []
            
        # Convert deque to list of dicts
        return [
            {"timestamp": ts, "value": val} 
            for ts, val in list(self.metrics[metric_type])[-limit:]
        ]
        
    def save_metrics(self, filename: Optional[str] = None) -> str:
        """
        Save metrics to a JSON file.
        
        Args:
            filename: Optional filename (datetime-based if not provided)
            
        Returns:
            Path to saved metrics file
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"metrics_{timestamp}.json"
            
        file_path = self.metrics_dir / filename
        
        # Create metrics snapshot
        metrics_data = self.get_metrics()
        
        # Add metadata
        metrics_data["metadata"] = {
            "timestamp": time.time(),
            "version": "1.0",
            "system": "oarc_rag"
        }
        
        # Save as JSON
        with open(file_path, 'w') as f:
            json.dump(metrics_data, f, indent=2)
            
        log.info(f"Saved metrics to: {file_path}")
        return str(file_path)
        
    def generate_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive performance report.
        
        Returns:
            Dict containing performance report data
        """
        metrics = self.get_metrics()
        
        # Calculate overall performance score (0-100)
        performance_score = self._calculate_performance_score()
        
        # Generate insights based on metrics
        insights = self._generate_performance_insights()
        
        # Create report
        report = {
            "timestamp": time.time(),
            "performance_score": performance_score,
            "insights": insights,
            "metrics_summary": {
                "latency": metrics["aggregated_metrics"].get("latency", {}),
                "retrieval_quality": metrics["aggregated_metrics"].get("retrieval_quality", {}),
                "cache_efficiency": metrics["aggregated_metrics"].get("cache_efficiency", {})
            },
            "operational_summary": {
                "uptime": metrics["operational_metrics"]["uptime"],
                "requests": metrics["operational_metrics"]["total_requests"],
                "error_rate": metrics["operational_metrics"]["error_rate"],
                "cache_hit_ratio": metrics["operational_metrics"]["cache_hit_ratio"]
            },
            "resource_summary": {
                "avg_memory": metrics["resource_metrics"].get("average_memory_usage", 0),
                "avg_cpu": metrics["resource_metrics"].get("average_cpu_usage", 0),
                "peak_memory": metrics["resource_metrics"].get("peak_memory_usage", 0),
                "peak_cpu": metrics["resource_metrics"].get("peak_cpu_usage", 0)
            },
            "alert_summary": {
                "total": len(metrics["alerts"]),
                "critical": len([a for a in metrics["alerts"] if a["level"] == "critical"]),
                "warning": len([a for a in metrics["alerts"] if a["level"] == "warning"]),
                "info": len([a for a in metrics["alerts"] if a["level"] == "info"])
            },
            "recommendations": self._generate_recommendations()
        }
        
        return report
        
    def _calculate_performance_score(self) -> float:
        """
        Calculate overall performance score from 0-100.
        
        Returns:
            Performance score
        """
        # Extract key metrics
        metrics = self.get_metrics()
        
        # Start with base score
        score = 100.0
        
        # Penalize for high latency
        latency = metrics["aggregated_metrics"].get("latency", {}).get("avg", 0)
        if latency > self.thresholds["latency_critical"]:
            score -= 30
        elif latency > self.thresholds["latency_warning"]:
            score -= 15
            
        # Penalize for low retrieval quality
        retrieval_quality = metrics["aggregated_metrics"].get("retrieval_quality", {}).get("avg", 1.0)
        if retrieval_quality < self.thresholds["similarity_warning"]:
            score -= 20
            
        # Penalize for errors
        error_rate = metrics["operational_metrics"].get("error_rate", 0)
        score -= error_rate * 100  # e.g., 0.05 error rate = -5 points
        
        # Penalize for resource usage
        memory_usage = metrics["resource_metrics"].get("average_memory_usage", 0)
        if memory_usage > self.thresholds["memory_critical"]:
            score -= 10
        elif memory_usage > self.thresholds["memory_warning"]:
            score -= 5
            
        # Limit the score range
        return max(0.0, min(100.0, score))
        
    def _generate_performance_insights(self) -> List[str]:
        """
        Generate insights based on metrics.
        
        Returns:
            List of insight strings
        """
        metrics = self.get_metrics()
        insights = []
        
        # Check latency trend
        latency_trend = metrics["aggregated_metrics"].get("latency", {}).get("trend")
        if latency_trend == "degrading":
            insights.append("Query latency is increasing, suggesting potential performance degradation.")
        elif latency_trend == "improving":
            insights.append("Query latency is decreasing, indicating performance improvement.")
            
        # Check retrieval quality
        retrieval_quality = metrics["aggregated_metrics"].get("retrieval_quality", {}).get("avg", 0)
        if retrieval_quality < self.thresholds["similarity_warning"]:
            insights.append("Retrieval quality is below target threshold, suggesting relevance issues.")
            
        # Check cache efficiency
        cache_hit_ratio = metrics["operational_metrics"].get("cache_hit_ratio", 0)
        if cache_hit_ratio < 0.3:
            insights.append("Low cache hit ratio indicates potential cache strategy improvements needed.")
        elif cache_hit_ratio > 0.8:
            insights.append("High cache hit ratio indicates effective cache utilization.")
            
        # Check operational mode transitions
        mode_transitions = metrics.get("operational_mode", {}).get("mode_transitions", 0)
        if mode_transitions > 10:
            insights.append("Frequent operational mode transitions may indicate system instability.")
            
        # Check alert patterns
        alert_count = len(self.alerts)
        if alert_count > 20:
            insights.append(f"High alert count ({alert_count}) suggests systemic issues requiring attention.")
            
        # Add placeholder insight if none generated
        if not insights:
            insights.append("System is operating within normal parameters.")
            
        return insights
        
    def _generate_recommendations(self) -> List[str]:
        """
        Generate recommendations based on metrics.
        
        Returns:
            List of recommendation strings
        """
        metrics = self.get_metrics()
        recommendations = []
        
        # Latency recommendations
        latency = metrics["aggregated_metrics"].get("latency", {}).get("avg", 0)
        if latency > self.thresholds["latency_warning"]:
            recommendations.append("Consider enabling PCA dimension reduction to improve query latency.")
            
        # Memory usage recommendations
        memory_usage = metrics["resource_metrics"].get("average_memory_usage", 0)
        if memory_usage > self.thresholds["memory_warning"]:
            recommendations.append("Consider vector quantization to reduce memory usage.")
            
        # Cache recommendations
        cache_hit_ratio = metrics["operational_metrics"].get("cache_hit_ratio", 0)
        if cache_hit_ratio < 0.4:
            recommendations.append("Review cache key generation strategy to improve hit rates.")
            
        # Vector database recommendations
        if self.database_metrics.get("chunks_stored", 0) > 10000:
            recommendations.append("Consider increasing HNSW ef_search parameter for better recall with large collections.")
            
        # Add placeholder recommendation if none generated
        if not recommendations:
            recommendations.append("No specific recommendations at this time. System is operating optimally.")
            
        return recommendations
        
    def _start_periodic_reporting(self) -> None:
        """Start background thread for periodic reporting."""
        def reporting_task():
            while self.enable_periodic_reporting:
                try:
                    report = self.generate_report()
                    self.save_metrics()
                    log.info(f"Periodic reporting - Performance score: {report['performance_score']:.1f}/100")
                except Exception as e:
                    log.error(f"Error in periodic reporting: {e}")
                finally:
                    time.sleep(self.reporting_interval)
                    
        self._reporting_thread = threading.Thread(
            target=reporting_task, 
            name="RAGMonitorReporting", 
            daemon=True
        )
        self._reporting_thread.start()
        
    def stop_periodic_reporting(self) -> None:
        """Stop periodic reporting."""
        self.enable_periodic_reporting = False
        if self._reporting_thread:
            self._reporting_thread.join(timeout=1.0)


# Create instance for easy access
monitor = RAGMonitor()
