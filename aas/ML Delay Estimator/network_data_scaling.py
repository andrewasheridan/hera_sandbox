import numpy as np

# data scaling
def delay_tx(delay, min_d = -0.040, max_d = 0.040):
    return (delay + abs(min_d)) / (abs(max_d) + abs(min_d))

def delay_itx(scaled_delay, min_d = -0.040, max_d = 0.040):
    return scaled_delay*(abs(max_d) + abs(min_d)) - abs(min_d)

def angle_tx(angles):
    return (angles + np.pi) / (2 * np.pi)

def angle_itx(scaled_angles):
    return scaled_angles*2.*np.pi - np.pi