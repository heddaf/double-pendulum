"""
===========================
Double Pendulum Animation
===========================

This animation illustrates the double pendulum problem
using the Euler's method.

Author: Hedda Forsman 4/25/2024
Credits: https://matplotlib.org/
"""

import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
from numpy import cos, sin

G = 9.8             # acceleration due to gravity, in m/s^2
L1 = 1.0            # length of pendulum 1 in m
L2 = 1.0            # length of pendulum 2 in m
L = L1 + L2         # total length of the combined pendulum
M1 = 1.0            # mass of pendulum 1 in kg
M2 = 1.0            # mass of pendulum 2 in kg
T_STOP = 10.0       # how many seconds to simulate
T_OFFSET = 0.01     # number of second increments
TH1 = 120.0         # initial angle θ1 (degrees)
TH2 = -10.0         # initial angle θ2 (degrees)
W1 = 0.0            # initial angular velocity w1 (degrees per second)
W2 = 0.0            # initial angular velocity w2 (degrees per second)

def derive(data):
    """Fills array of derivatives based on input data array

    Parameters
    ----------
    data : float array
        data[0] => angle for θ1
        data[1] => angle velocity w1
        data[2] => angle for θ2
        data[3] => angle velocity w2

    Returns
    -------
    float array:
        [0] => angle velocity w1
        [1] => calculated value
        [2] => angle velocity w2
        [3] => calculated value
    """

    dydx = np.zeros_like(data)

    dydx[0] = data[1]

    delta = data[2] - data[0]

    # Euler's method:
    denominator_1 = (M1+M2) * L1 - M2 * L1 * cos(delta)**2
    dydx[1] = ((M2 * L1 * data[1]**2 * sin(delta) * cos(delta)
                + M2 * G * sin(data[2]) * cos(delta)
                + M2 * L2 * data[3]**2 * sin(delta)
                - (M1+M2) * G * sin(data[0]))
               / denominator_1)

    dydx[2] = data[3]

    denominator_2 = (L2/L1) * denominator_1
    dydx[3] = ((- M2 * L2 * data[3]**2 * sin(delta) * cos(delta)
                + (M1+M2) * G * sin(data[0]) * cos(delta)
                - (M1+M2) * L1 * data[1]**2 * sin(delta)
                - (M1+M2) * G * sin(data[2]))
               / denominator_2)
    # breakpoint()
    # Ω θ
    return dydx

# create time data in the form of an array from 0 to T_STOP
time_values = np.arange(0, T_STOP, T_OFFSET)

# set initial state according to constants and covert degrees to radians
state = np.radians([TH1, W1, TH2, W2])

# create empty two-dimensional array matching the length of the time values
y = np.empty((len(time_values), 4))

# set the first item in the derivatives array to the initial state
y[0] = state

# iterate over time array and calculate derivatives
for i in range(1, len(time_values)):
    y[i] = y[i - 1] + derive(y[i - 1]) * T_OFFSET

# grab all values at index 0 in the 2-dimensional array and calculate
# (see https://numpy.org/doc/stable/reference/arrays.ndarray.html)
x1 = L1*sin(y[:, 0])
y1 = -L1*cos(y[:, 0])

# grab all values at index 2 in the 2-dimensional array and calculate
x2 = L2*sin(y[:, 2]) + x1
y2 = -L2*cos(y[:, 2]) + y1

# set up animation
fig = plt.figure(figsize=(5, 4))
ax = fig.add_subplot(autoscale_on=False, xlim=(-L, L), ylim=(-L, 1.))
ax.set_aspect('equal')
ax.grid()

line, = ax.plot([], [], 'o-', lw=2)
trace, = ax.plot([], [], '.-', lw=1, ms=2)
TIME_TEMPLATE = 'time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)


def animate(idx):
    """Sets the line, trace and time_text on the plot

    Parameters
    ----------
    idx : int
        Item index in the data arrays

    Returns
    -------
    line, trace, time_text objects
    """

    thisx = [0, x1[idx], x2[idx]]
    thisy = [0, y1[idx], y2[idx]]

    history_x = x2[:idx]
    history_y = y2[:idx]

    line.set_data(thisx, thisy)
    trace.set_data(history_x, history_y)
    time_text.set_text(TIME_TEMPLATE % (idx*T_OFFSET))
    return line, trace, time_text

# animate!
ani = animation.FuncAnimation(fig, animate, len(y), interval=T_OFFSET*1000, blit=True)
plt.show()
