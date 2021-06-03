try:
    from .visualization import VtkWindow
    from .inout import *
    from .measure import *
except:
    from visualization import VtkWindow
    from inout import *
    from measure import *

import sys
import traceback
from multiprocessing.pool import ThreadPool
from pathlib import Path
import time
import matplotlib.pyplot as plt

path = Path('E:/E_Documents/Research/Computer Vision Collaboration/Erica Lilleodden/fib serial section data.tif')
im = read_tiff_stack(path)


def start_skeletonize(arg):
    print('hello')
    return skeletonize(im)

def finish_skeletonize(result):
    print('done')
    plt.imshow(result[0][20,:,:])
    plt.show()

pool = ThreadPool(2)
pool.map_async(start_skeletonize, [1],
               callback=finish_skeletonize)

time.sleep(100)
