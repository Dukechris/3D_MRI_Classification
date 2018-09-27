from __future__ import division, print_function

import os
import glob
import re
import nibabel as nib

import numpy as np
from math import ceil
import threading

from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

# from keras import utils
from keras.utils import np_utils



class SubjectData(object):
    """Data directory structure:
    Data/SUB_DIR_FOR_EACH_CLASSES
    wADNI_002_S_0619_MR_MPR-R__GradWarp__N3__Scaled_Br_20070411125458928_S15145_I48617.nii
    wADNI_002_S_0816_MR_MPR__GradWarp__B1_Correction__N3__Scaled_2_Br_20081002102135862_S18402_I118984.nii
    ...
    """
    def __init__(self, directory):
        self.directory = os.path.normpath(directory)
        # load all data(images and labels)into memory
        self.loadImages()

    @property
    def images(self):
        return self.all_images

    def loadImages(self):
        glob_search = os.path.join(self.directory, "w*.nii")
        nii_files = sorted(glob.glob(glob_search))
        self.all_images = []

        for nii_file in nii_files:
            image = nib.load(nii_file)
            image_data = np.nan_to_num(image.get_fdata())
            self.all_images.append(image_data)
        self.image_height, self.image_width, self.image_length = image_data.shape



def load_images(data_dir):
    """Load all subject images and labels from data directory. The directories and images are read in sorted order.
    Arguments:
      data_dir - path to data directory (TrainingSet, Test1Set or Test2Set)
    Output:
      tuples of (images, labels), 
      images is 5-d tensors of shape
      labels is a list
      (batchsize, height, width, length, channels). 
      dtype of images is float64
      labels is a list of strings.
    """
    #### sub dir for each class
    glob_search = os.path.join(data_dir + '/*')
    subject_dirs = sorted(glob.glob(glob_search))
    if len(subject_dirs) == 0:
        raise Exception("No subject directors found in {}".format(data_dir))
    #### sub dir for each class
    # load all images into memory (dataset is small)
    images = []
    labels = []
    i = 0
    for subject_dir in subject_dirs:
        #### for every classes, done
        p = SubjectData(subject_dir)   # generate p as cur class
        images += p.images  # append image data
        cur_labels = [i] * len(p.images)
        i+=1
        labels += cur_labels  # append labels
        
    # reshape to account for channel dimension
    images = np.asarray(images)[:,:,:,:,None]
    labels = np_utils.to_categorical(labels, 2)
    # reset str labels to 0 or 1 labels
    # labels = np.asarray(labels) # labels is a list
    return images, labels





class Iterator(object):
    """Abstract base class for image data iterators.

    # Arguments
        n: Integer, total number of samples in the dataset to loop over.
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seeding for data shuffling.
    """
    def __init__(self, n, batch_size, shuffle, seed):
        self.n = n
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batch_index = 0
        self.total_batches_seen = 0
        self.lock = threading.Lock()
        self.index_generator = self._flow_index(n, batch_size, shuffle, seed)

    def reset(self):
        self.batch_index = 0

    def _flow_index(self, n, batch_size=32, shuffle=False, seed=None):
        # Ensure self.batch_index is 0.
        self.reset()
        while 1:
            if seed is not None:
                np.random.seed(seed + self.total_batches_seen)
            if self.batch_index == 0:
                index_array = np.arange(n)
                if shuffle:
                    index_array = np.random.permutation(n)

            current_index = (self.batch_index * batch_size) % n
            if n > current_index + batch_size:
                current_batch_size = batch_size
                self.batch_index += 1
            else:
                current_batch_size = n - current_index
                self.batch_index = 0
            self.total_batches_seen += 1
            yield (index_array[current_index: current_index + current_batch_size],
                   current_index, current_batch_size)

    def __iter__(self):
        # Needed if we want to do something like:
        # for x, y in data_gen.flow(...):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)

    def next(self):
        """For python 2.x.

        # Returns
            The next batch.
        """
        # Keeps under lock only the mechanism which advances the indexing of each batch.
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        batch_x = np.zeros(tuple([current_batch_size] + list(self.x.shape)[1:]), dtype='float64')
        for i, j in enumerate(index_array):
            x = self.x[j]
#             x = self.image_data_generator.random_transform(x.astype(K.floatx()))
#             x = self.image_data_generator.standardize(x)
            batch_x[i] = x
    
        if self.y is None:
            return batch_x
        batch_y = self.y[index_array]
        return batch_x, batch_y






class NumpyArrayIterator(Iterator):
    """Iterator yielding data from a Numpy array.

    # Arguments
        x: Numpy array of input data.
        y: Numpy array of targets data.
        image_data_generator: Instance of `ImageDataGenerator`
            to use for random transformations and normalization.
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seed for data shuffling.
        data_format: String, one of `channels_first`, `channels_last`.
        save_to_dir: Optional directory where to save the pictures
            being yielded, in a viewable format. This is useful
            for visualizing the random transformations being
            applied, for debugging purposes.
        save_prefix: String prefix to use for saving sample
            images (if `save_to_dir` is set).
        save_format: Format to use for saving sample images
            (if `save_to_dir` is set).
    """

    def __init__(self, x, y, batch_size=32, shuffle=False, seed=None, data_format='channels_last'):
        if y is not None and len(x) != len(y):
            raise ValueError('X (images tensor) and y (labels) '
                             'should have the same length. '
                             'Found: X.shape = %s, y.shape = %s' %
                             (np.asarray(x).shape, np.asarray(y).shape))

        self.x = np.asarray(x, dtype='float64')

        if self.x.ndim != 5:
            raise ValueError('Input data in `NumpyArrayIterator` '
                             'should have rank 5. You passed an array '
                             'with shape', self.x.shape)
        channels_axis = 4 if data_format == 'channels_last' else 1
        # x.shape = [n.samples, x_dim, y_dim, z_dim, channels]
        # where x.shape[0] = n.samples
        if self.x.shape[channels_axis] not in {1, 3, 4}:
            raise ValueError('NumpyArrayIterator is set to use the '
                             'data format convention "' + data_format + '" '
                             '(channels on axis ' + str(channels_axis) + '), i.e. expected '
                             'either 1, 3 or 4 channels on axis ' + str(channels_axis) + '. '
                             'However, it was passed an array with shape ' + str(self.x.shape) +
                             ' (' + str(self.x.shape[channels_axis]) + ' channels).')
        if y is not None:
            self.y = np.asarray(y)
        else:
            self.y = None
            
        self.data_format = data_format
#         self.save_to_dir = save_to_dir
#         self.save_prefix = save_prefix
#         self.save_format = save_format
        super(NumpyArrayIterator, self).__init__(x.shape[0], batch_size, shuffle, seed)




# Revised ImageDataGenerator For 3D Images 
# --- ImageGenerator
class ImageGenerator(object):
    def __init__(self):
        # print('Call ImageGenerator!')
        self.data_format = 'channels_last'
        if self.data_format == 'channels_first':
            self.channel_axis = 1
        if self.data_format == 'channels_last':
            self.channel_axis = 4

    def flow(self,x, y=None, batch_size=32, shuffle=True, seed=None): 
        return NumpyArrayIterator(x, y, batch_size=batch_size, shuffle=shuffle, seed=seed, data_format='channels_last')


def create_generators(data_dir, batch_size, validation_split=0.0,
                      shuffle_train_val=True, shuffle=True, seed=None):
    images, labels = load_images(data_dir)

    # before: type(labels) = list and type(images) = float32
    # convert images to double-precision
    images = images.astype('float64')
    # convert labels to int
    # labels = (np.asarray(labels) == 'NL').astype('int32')


    if seed is not None:
        np.random.seed(seed) # random seed

    if shuffle_train_val:
        # shuffle images and labels in parallel
        rng_state = np.random.get_state()
        np.random.shuffle(images)
        np.random.set_state(rng_state)
        np.random.shuffle(labels)

    # split out last %(validation_split) of images as validation set
    split_index = int((1-validation_split) * len(images))
    
    # produce generator 
    idg = ImageGenerator()
    train_generator = idg.flow(images[:split_index], labels[:split_index], # split_index
                                   batch_size=batch_size, shuffle=shuffle) 

    train_steps_per_epoch = ceil(split_index / batch_size)

    if validation_split > 0.0:
        idg = ImageGenerator()
        val_generator = idg.flow(images[split_index:], labels[split_index:],
                                     batch_size=batch_size, shuffle=shuffle)
    else:
        val_generator = None

    val_steps_per_epoch = ceil((len(images) - split_index) / batch_size)

    return (train_generator, train_steps_per_epoch, val_generator, val_steps_per_epoch)





