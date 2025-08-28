import numpy as np

class MetricsCollector:
    def __init__(self):
        self.total_requests = 0
        self.successful = 0
        self.failed = 0
        self.cold_starts = 0
        self.instance_counts = []
        self.utilizations = []
        self.latencies = []  # per-request latencies in seconds

    def add_tick(self, reqs, succ, fail, cold, instance_count, util, latencies_in_tick):
        self.total_requests += reqs
        self.successful += succ
        self.failed += fail
        self.cold_starts += cold
        self.instance_counts.append(instance_count)
        self.utilizations.append(util)
        if latencies_in_tick:
            self.latencies.extend(latencies_in_tick)

    def summary(self):
        summary = {
            'total_requests': self.total_requests,
            'successful': self.successful,
            'failed': self.failed,
            'cold_starts': self.cold_starts,
            'avg_instances': float(np.mean(self.instance_counts)) if self.instance_counts else 0.0,
            'avg_utilization': float(np.mean(self.utilizations)) if self.utilizations else 0.0,
        }
        if self.latencies:
            summary['p50'] = float(np.percentile(self.latencies, 50))
            summary['p95'] = float(np.percentile(self.latencies, 95))
            summary['p99'] = float(np.percentile(self.latencies, 99))
        else:
            summary['p50'] = summary['p95'] = summary['p99'] = None
        return summary
