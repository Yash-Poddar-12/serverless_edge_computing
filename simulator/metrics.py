import numpy as np

class MetricsCollector:
    def __init__(self):
        self.total_requests = 0
        self.successful = 0
        self.failed = 0
        self.cold_starts = 0
        self.instance_counts = []
        self.utilizations = []

    def add_tick(self, reqs, succ, fail, cold, instance_count, util):
        self.total_requests += reqs
        self.successful += succ
        self.failed += fail
        self.cold_starts += cold
        self.instance_counts.append(instance_count)
        self.utilizations.append(util)

    def summary(self):
        return {
            'total_requests': self.total_requests,
            'successful': self.successful,
            'failed': self.failed,
            'cold_starts': self.cold_starts,
            'avg_instances': np.mean(self.instance_counts) if self.instance_counts else 0,
            'avg_utilization': np.mean(self.utilizations) if self.utilizations else 0,
        }
