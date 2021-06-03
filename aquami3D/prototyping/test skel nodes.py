import numpy as np
from skimage import morphology
import matplotlib.pyplot as plt


a = np.array(
    [[0,1,1,1,0,0,0,1,1,1,0],
     [0,1,1,1,0,0,0,1,1,1,0],
     [0,0,1,1,1,0,1,1,1,0,0],
     [0,0,0,1,1,1,1,1,0,0,0],
     [0,0,0,0,1,1,1,0,0,0,0],
     [0,0,0,0,1,1,1,0,0,0,0],
     [0,0,0,0,1,1,1,0,0,0,0],
     [0,0,0,1,1,1,1,1,0,0,0],
     [0,0,1,1,1,0,1,1,1,0,0],
     [0,1,1,1,0,0,0,1,1,1,0],
     [0,1,1,1,0,0,0,1,1,1,0]])

skel = morphology.skeletonize(a).astype('bool')

b = np.zeros(a.shape)
b[a==1] = 125
b[skel==1] = 255

plt.imshow(b, cmap=plt.cm.gray, interpolation=None)
plt.show()


