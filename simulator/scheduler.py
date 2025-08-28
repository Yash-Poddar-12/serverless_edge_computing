# Warmest scheduler and other options
from collections import deque
import random

class Scheduler:
    def __init__(self, policy='warmest'):
        self.policy = policy
        # maintain list of instances externally; scheduler uses instance list

    def pick_instance(self, instances, now):
        if not instances:
            return None
        if self.policy == 'random':
            return random.choice(instances)
        if self.policy == 'round_robin':
            # simple round robin using deque rotation
            d = deque(instances)
            d.rotate(-1)
            return d[0]
        # warmest: pick instance with most recent last_used_time (i.e., max last_used_time)
        if self.policy == 'warmest':
            # prefer warm instances first
            warmed = [inst for inst in instances if inst.is_warm(now)]
            if warmed:
                # warmest = highest last_used_time (most recently used)
                warmed_sorted = sorted(warmed, key=lambda i: (i.last_used_time or -1), reverse=True)
                return warmed_sorted[0]
            # fallback: least recently used (coldest)
            sorted_all = sorted(instances, key=lambda i: (i.last_used_time or -1))
            return sorted_all[0]
        # default fallback
        return random.choice(instances)
