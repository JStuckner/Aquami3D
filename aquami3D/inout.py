import numpy as np
from skimage import io
from skimage.morphology import binary_dilation

def read_tiff_stack(file_path):
    im = io.imread(file_path)
    im = im > 0
    im = np.swapaxes(im, 0, 2)
    return im

def read_xyz(file_path):
    import time # OVITO loads about 35% faster than this
    t = time.time()

    # puts the data in a numpy array really fast (25% slower than OVITO)
    # https://stackoverflow.com/questions/8956832/python-out-of-memory-on-large-csv-file-numpy?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
    def iter_loadtxt(filename, delimiter=' ', skiprows=0, skipcols=0,
                     dtype=float):
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

    # Find the column start
    with(open(file_path, 'r')) as f:
        next(f) #skip first line.
        meta = f.readline() # read the second line.
        meta = meta.split('Properties=')[1] # get properties.
        meta = meta.split('pos')[0] # get columns before pos.
        meta = meta.split(':') # separate keys and values.
        skipcols = sum([int(i) for i in meta[2::3]]) # add columns before pos.

    data = iter_loadtxt(file_path, skiprows=2, skipcols=skipcols) # get data
    data = data * 0.407 # scale lattice parameter to 1 pixel
    data = np.rint(data).astype(int) # convert to integer
    data = data - data.min(axis=0) # set minimum to 0 for all axes
    volume = np.zeros(data.max(axis=0) + 1, dtype='bool') # create array
    volume[data[:, 0], data[:, 1], data[:, 2]] = 1 # fill from coordinates
    # each atom is more than just a point
    volume = binary_dilation(volume, selem=np.ones((2, 2, 2)))
    print(time.time()-t)
    return volume


def save_measurements_to_text(file, title, list):
    with open(file, 'wb') as f:
        f.write(title)
        f.write('\n'.join(list))


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    fpath = r'E:\E_Documents\Research\Computer Vision Collaboration\Erica Lilleodden/indentor dump.pov'
    a = read_xyz(fpath)
    plt.imshow(a[:,:,200])
    plt.show()
