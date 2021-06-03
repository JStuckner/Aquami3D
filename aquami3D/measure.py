import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology, transform
from skimage.measure import label
from scipy import ndimage
from scipy.signal import convolve


def resize_and_get_pixel_size(im, x_pixel_size, y_pixel_size, z_pixel_size):
    rows, cols, slices = im.shape
    # Get the pixel size of the longest direction.  The other directions
    # will be scaled to the same size.
    pixel_size = max(x_pixel_size, y_pixel_size, z_pixel_size)
    scale = (pixel_size / x_pixel_size,
             pixel_size / y_pixel_size,
             pixel_size / z_pixel_size)
    output_shape = (int(round(rows*scale[0],0)),
                    int(round(cols*scale[1],0)),
                    int(round(slices*scale[2],0)))
    scaled_im = transform.resize(im, output_shape)
    return scaled_im, pixel_size



def skeletonize(im):
    skel = morphology.skeletonize_3d(im).astype('bool')
    return(skel)


def distance_transform(im):
    dt = ndimage.morphology.distance_transform_edt(im)
    return dt


def calculate_diameter(skel, dt, pixel_size):
    diams = skel * dt * 2 / skel.max()
    av = int(np.average(diams[diams>0]))
    av *= 2 # exclude edge pixels that are twice the average
    diams = diams[av:-av, av:-av, av:-av]
    diams = diams[diams>2]
    diams = diams * pixel_size
    return diams


def count_neighbors(skel):
    strel = np.ones((3, 3, 3))
    strel[1, 1, 1] = 0
    conv = convolve(skel, strel, mode='same')
    neighbors = np.round((conv * skel)).astype(np.uint8)
    return neighbors


def find_nodes(neighbors):
    nodes = neighbors > 2
    return nodes


def label_ligaments(neighbors):
    ligaments = neighbors < 3
    ligaments[neighbors == 0] = 0
    labels = label(ligaments, connectivity=3)
    return labels


def calculate_lengths(labels, pixel_size):
    lengths = np.bincount(labels.flatten())
    lengths[0] = 0 # filter out background
    lengths = lengths[lengths>0]
    lengths = lengths * pixel_size
    return lengths

def remove_terminal_ligaments(labels, neighbors, return_terminal=False):
    terminals = neighbors == 1
    term_labels = np.unique(terminals * labels)
    terminal_ligaments = np.isin(labels, term_labels[1:])
    terminal_removed = (neighbors > 1) ^ terminal_ligaments
    if return_terminal:
        return terminal_removed, terminal_ligaments
    else:
        return terminal_removed


def fnm2(im, skel, nodes, labels, dist):
    node_mask = np.zeros(im.shape, dtype='bool')
    node_dialate = nodes > 0
    while True:
        node_dialate = morphology.binary_dilation(node_dialate)
        node_labels = morphology.label(node_dialate)
        inside_node_labels = node_labels * im
        unique, counts = np.unique(node_labels, return_counts=True)
        _, counts_inside = np.unique(inside_node_labels, return_counts=True)
        print(unique, counts, counts_inside)
        for u, c, ci in zip(unique[1:], counts[1:], counts_inside[1:]):
            if c > ci:
                node_mask[node_labels == u] = 1
                node_dialate[node_labels == u] = 0
        if np.count_nonzero(node_dialate) == 0:
            break
    return(node_mask * im)


def fnm3(im, skel, nodes, labels, dist):
    node_mask = np.zeros(im.shape, dtype='bool')
    xx, yy, zz = node_mask.shape
    node_dist = nodes * dist
    idx = np.nonzero(nodes)
    #print(len(idx[0]))
    for x, y, z in zip(idx[0], idx[1], idx[2]):
        print(x,y,z)
        radius = node_dist[x, y, z]
        n = int(2 * radius + 1)
        Z,Y,X = np.mgrid[-x:xx-x, -y:yy-y, -z:zz-z]
        mask = X ** 2 + Y ** 2 + Z ** 2 <= radius ** 2
        #print(mask.shape)
        node_mask[mask] = 1
        #array = np.zeros((n, n), dtype='bool')
        #array[mask] = 1
        #node_mask[x-X:x+X+1, y-Y:y+Y+1, z-Z:z+Z+1] = 1
    return(node_mask)



def find_node_mask(im, skel, nodes, labels, distance_transform):
    node_mask = np.zeros(im.shape, dtype='bool')
    node_dist = (nodes * distance_transform).astype(np.uint8)
    node_dist_list = np.unique(node_dist).tolist()
    #print(node_dist_list)
    spheres = SphereLookupTable(node_dist_list)
    for dist in node_dist_list[1:]:
        this_dist = np.zeros(im.shape)
        this_dist[node_dist==dist] = 1
        this_mask = spheres.get_mask(dist)
        conv = convolve(this_dist,
                        this_mask,
                        mode='same')
        conv = conv > 0.5
        node_mask[conv==1] = 1

    # spheres = SphereLookupTable(range(10))
    # print(spheres.get_mask(2))
    # test = np.zeros((7,7,7))
    # test[3,3,3] = 1
    # test = convolve(test, spheres.get_mask(node_dist_list[3]), mode='same')
    # test = test > 0.5

    return node_mask


class SphereLookupTable(object):
    # Generates a dictionary of sphere structuring elements.
    # This way the structuring elements only have to be generated once.
    table = {}

    def __init__(self, radii):
        for i in radii:
            self.table[i] = self.generate_mask(i)

    def generate_mask(self, r):
        return morphology.ball(r, dtype='bool')

    def get_mask(self, r):
        return self.table[r]


class VolumeData(object):

    def __init__(self, im, pixel_size = 1.0, status=None, progress=None,
                 display=None, plot=None, invert_im = False):
        self.pixel_size = pixel_size

        # widget control
        self.status = status
        self.progress = progress
        self.display = display
        self.plot = plot

        # volume data
        self.im = im < 1 if invert_im else im > 0# Volume data
        self.skel = None #binary mask locating the backbone
        self.nodes = None #binary mask locating the nodes
        self.terminal = None # binary mask locating terminal ligaments
        self.node_mask = None
        self.lengths = None
        self.all_diameters = None
        self.terminal_diameters = None
        self.node_diameters = None
        self.connected_diameters = None
        self.percent_terminal = None

        # data properties
        self.shape = self.im.shape

    def calculate(self):
        if self.progress is not None:
            self.progress.setVisible(True)

        # Skeletonize
        self.update_progress('Skeletonizing...', 1)
        self.skel = skeletonize(self.im)

        # Distance
        self.update_progress('Calculating distance transform...', 20)
        distance = distance_transform(self.im)

        # Find number of neighbors
        self.update_progress('Counting pixel neighbors...', 25)
        neighbors = count_neighbors(self.skel)

        # Find nodes
        self.update_progress('Finding nodes...', 30)
        self.nodes = find_nodes(neighbors)

        # Label ligaments
        self.update_progress('Separating ligaments...', 40)
        labels = label_ligaments(neighbors)

        # Remove terminal ligaments
        self.update_progress('Finding terminal ligaments...', 50)
        skel_without_term, self.terminal = remove_terminal_ligaments(
            labels, neighbors, True)

        # Find node mask
        #self.update_progress('Finding node mask...', 60)
        #self.node_mask =  fnm3(self.im, self.skel, self.nodes, labels,
        #                            distance)

        # Calculate lengths
        self.update_progress('Calculating length...', 60)
        self.lengths = calculate_lengths(labels * skel_without_term, self.pixel_size)
        #self.plot.plot(self.lengths, 'length [pixels]')

        # Diameter
        self.update_progress('Calculating diameter...', 70)
        self.all_diameters = calculate_diameter(self.skel, distance, self.pixel_size)
        if self.plot is not None:
            self.plot.plot(self.all_diameters, 'diameter [units]')
        self.terminal_diameters = calculate_diameter(self.terminal, distance, self.pixel_size)
        self.node_diameters = calculate_diameter(self.nodes, distance, self.pixel_size)
        self.connected_diameters = calculate_diameter(skel_without_term, distance, self.pixel_size)
        self.percent_terminal = 100* len(self.terminal_diameters) / len(self.all_diameters)

        # Finish
        self.update_progress('', 100)

    def export(self, path):
        with open(path, 'w') as f:
            f.write('Average of all ligament diameters: ')
            f.write(str(round(np.mean(self.all_diameters),2)))
            f.write('\n')
            f.write('Average connected ligament diameter: ')
            f.write(str(round(np.mean(self.connected_diameters), 2)))
            f.write('\n')
            f.write('Average terminal ligament diameter: ')
            f.write(str(round(np.mean(self.terminal_diameters), 2)))
            f.write('\n')
            f.write('Average node point diameter: ')
            f.write(str(round(np.mean(self.node_diameters), 2)))
            f.write('\n')
            f.write('Average ligament length (node to node) (NOT ACCURATE): ')
            f.write(str(round(np.mean(self.lengths), 2)))
            f.write('\n')
            f.write('Percent terminal ligaments (linear length): ')
            f.write(str(round(np.mean(self.percent_terminal), 2)))
            f.write('%')
            f.write('\n\n\n')

            f.write('All diameter measurements')
            f.write('\n')
            f.write('\n'.join([str(i) for i in self.all_diameters]))
            f.write('\n\n')
            f.write('Connected ligament diameter measurements')
            f.write('\n')
            f.write('\n'.join([str(i) for i in self.connected_diameters]))
            f.write('\n\n')
            f.write('Terminal ligament diameter measurements')
            f.write('\n')
            f.write('\n'.join([str(i) for i in self.terminal_diameters]))
            f.write('\n\n')
            f.write('Node point diameter measurements')
            f.write('\n')
            f.write('\n'.join([str(i) for i in self.node_diameters]))
            f.write('\n\n')
            f.write('Ligament length measurements')
            f.write('\n')
            f.write('\n'.join([str(i) for i in self.lengths]))
            f.write('\n\n')

    def update_progress(self, message, value):
        if self.status is not None:
            self.status.setText(message)
        else:
            print(message)
        if self.progress is not None: self.progress.setValue(value)
        if self.progress is not None and value == 100:
            self.progress.setVisible(False)
        if self.display is not None:
            self.display([0, self.shape[0],
                          0, self.shape[1],
                          0, self.shape[2]])


if __name__ == '__main__':
    from inout import read_tiff_stack
    fpath = (r'E:\E_Documents\Research\Computer Vision Collaboration\Erica '
             r'Lilleodden/fib serial section data.tif')
    im = read_tiff_stack(fpath)
    #im = im[:100, :100, :100]
    volume = VolumeData(im, 0.1)
    volume.calculate()
    opath = (r'E:\E_Documents\Research\Computer Vision Collaboration\Erica '
             r'Lilleodden/data.txt')
    volume.export(opath)
    #a = SphereLookupTable(10)
    #print(a.get_mask(3))
    #print('start')
    #im, pixel_size = resize_and_get_pixel_size(im, 1,1,2)
