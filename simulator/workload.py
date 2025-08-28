from .config import ARRIVAL_LAMBDA
from .utils import poisson_arrivals, sample_service_time

class WorkloadGenerator:
    def __init__(self, lmbda=ARRIVAL_LAMBDA):
        self.lmbda = lmbda

    def arrivals_for_tick(self):
        n = poisson_arrivals(self.lmbda)
        # return list of job durations (seconds)
        return [sample_service_time() for _ in range(n)]
