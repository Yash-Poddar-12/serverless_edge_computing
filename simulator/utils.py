import random
import math
import numpy as np

def poisson_arrivals(lmbda):
    # Return number of arrivals in a 1-second tick using Poisson
    return np.random.poisson(lmbda)

def sample_service_time():
    # sample according to mix: short/med/long
    r = random.random()
    if r < 0.90:
        return random.uniform(0.1, 1.0)
    elif r < 0.99:
        return random.uniform(1.0, 10.0)
    else:
        return random.uniform(10.0, 100.0)
