import numpy as np
import time

#https://stackoverflow.com/questions/8956832/python-out-of-memory-on-large-csv-file-numpy?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
def iter_loadtxt(filename, delimiter=' ', skiprows=0, skipcols=0, dtype=float):
    def iter_func():
        with open(filename, 'r') as infile:
            for _ in range(skiprows):
                next(infile)
            for line in infile:
                line = line.rstrip().split(delimiter)[skipcols:skipcols+3]
                for item in line:
                    yield dtype(item)
        iter_loadtxt.rowlength = len(line)
    data = np.fromiter(iter_func(), dtype=dtype)
    data = data.reshape((-1, iter_loadtxt.rowlength))
    return data


t0 = time.time()
path = r'E:\E_Documents\Research\Computer Vision Collaboration\Erica Lilleodden/indentor dump.pov'
# OVITO takes 17 seconds to load this file
data = iter_loadtxt(path, skiprows=2, skipcols=2)
data = data.astype(int)
data = data - data.min(axis=0)
a = np.zeros(data.max(axis=0)+1, dtype='bool')
a[data[:,0], data[:,1], data[:,2]] = 1
print('Time: ', time.time()-t0)