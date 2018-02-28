

import sys
from math import sin, cos, radians

import numpy as np     # installed with matplotlib
import matplotlib.pyplot as plt
from math import radians

# Create a string with spaces proportional to a cosine of x in degrees
def make_dot_string(x):
    rad = radians(x)                             # cos works with radians
    numspaces = int(20 * cos(rad) + 20)   # scale to 0-40 spaces
    str = '|' * numspaces + 'o'                  # place 'o' after the spaces
    return str

def print_sin():
    for i in range(0, 1800, 12):
        s = make_dot_string(i)
        print(s)



def main():
    x = np.arange(0, radians(180), radians(20))
    plt.plot(x, np.cos(x), 'b')
    plt.plot(x, x, 'r')
    plt.show()

main()