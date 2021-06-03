from scipy.signal import convolve
import numpy as np

a = np.array([[[1, 0, 0, 0],
               [0, 1, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0]],
              [[0, 0, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 1, 0],
               [0, 0, 0, 0]],
              [[0, 0, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 1]]
              ])

strel = np.ones((3, 3, 3))
strel[1, 1, 1] = 0
conv = convolve(a, strel, mode='same')
conv = a * conv

print(conv)