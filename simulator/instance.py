import time

class FunctionInstance:
    def __init__(self, created_time, warm_time):
        self.created_time = created_time
        self.warm_time = warm_time
        self.last_used_time = None
        self.busy_until = created_time  # free at creation
        self.total_busy = 0.0

    def is_warm(self, now):
        if self.last_used_time is None:
            # if never used, warmness is based on creation time
            return (now - self.created_time) <= self.warm_time
        return (now - self.last_used_time) <= self.warm_time

    def assign(self, now, duration):
        start = max(now, self.busy_until)
        self.busy_until = start + duration
        self.last_used_time = start
        self.total_busy += duration
        return start
