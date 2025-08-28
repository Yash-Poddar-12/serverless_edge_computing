# simulator/config.py
# --- Simulation core parameters ---
SIM_SECONDS_PER_EPISODE = 60 * 60   # seconds per episode (1 hour)
TIME_STEP = 1                       # simulation tick in seconds
MAX_INSTANCES = 50                  # hard cap on instances

# --- Workload / service time distribution ---
WORKLOAD_SHORT_RATIO = 0.90
WORKLOAD_MED_RATIO = 0.09
WORKLOAD_LONG_RATIO = 0.01

SHORT_MIN, SHORT_MAX = 0.1, 1.0
MED_MIN, MED_MAX = 1.0, 10.0
LONG_MIN, LONG_MAX = 10.0, 100.0

# --- Warm-time (seconds) ---
DEFAULT_WARM_TIME = 180

# --- Q-learning / RL params ---
Q_ALPHA = 0.1
Q_GAMMA = 0.95
Q_EPSILON_START = 0.9
Q_EPSILON_MIN = 0.01
Q_EPSILON_DECAY = 0.995

# --- Reward weights (paper baseline) ---
W1, W2, W3, W4 = 0.4, 0.2, 0.2, 0.2

# --- Arrival rate (requests per second) ---
ARRIVAL_LAMBDA = 0.5
