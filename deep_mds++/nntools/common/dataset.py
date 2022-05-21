"""Data fetching
"""
# MIT License
# 
# Copyright (c) 2018 Yichun Shi
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import sys
import os
import time
import math
import random
import shutil
from multiprocessing import Process, Queue

import h5py
import numpy as np

class DataClass(object):
    def __init__(self, class_name, indices, label):
        self.class_name = class_name
        self.indices = list(indices)
        self.label = label
        return

    def random_pair(self):
        return np.random.permutation(self.indices)[:2]

    def random_samples(self, num_samples_per_class, exception=None):
        indices_temp = self.indices[:]
        if exception is not None:
            indices_temp.remove(exception)
        # Sample indices multiple times when more samples are required than present.
        indices = []
        iterations = int(np.ceil(1.0*num_samples_per_class / len(indices_temp)))
        for i in range(iterations):
            sample_indices = np.random.permutation(indices_temp)
            indices.append(sample_indices)
        indices = np.concatenate(indices, axis=0)[:num_samples_per_class]
        return indices

    def build_clusters(self, cluster_size):
        permut_indices = np.random.permutation(self.indices)
        cutoff = (permut_indices.size // cluster_size) * cluster_size
        clusters = np.reshape(permut_indices[:cutoff], [-1, cluster_size])
        clusters = list(clusters)
        if permut_indices.size > cutoff:
            last_cluster = permut_indices[cutoff:]
            clusters.append(last_cluster)
        return clusters




class Dataset():

    def __init__(self, path=None):
        self.DataClass = DataClass
        self.num_classes = None
        self.classes = None
        self.images = None
        self.labels = None
        self.minutiae_maps = None
        self.features = None
        self.index_queue = None
        self.queue_idx = None
        self.batch_queue = None

        if path is not None:
            self.init_from_path(path)

    def clear(self):
        del self.classes
        self.__init__()

    def init_from_path(self, path):
        path = os.path.expanduser(path)
        _, ext = os.path.splitext(path)
        if os.path.isdir(path):
            self.init_from_folder(path)
        elif ext == '.txt':
            self.init_from_list(path)
        elif ext == '.hdf5':
            self.init_from_hdf5(path)
        else:
            raise ValueError('Cannot initialize dataset from path: %s\n\
                It should be either a folder, .txt or .hdf5 file' % path)
        print('%d images of %d classes loaded' % (len(self.images), self.num_classes))

    def init_from_folder(self, folder):
        folder = os.path.expanduser(folder)
        class_names = os.listdir(folder)
        class_names.sort()
        images = []
        labels = []
        for label, class_name in enumerate(class_names):
            classdir = os.path.join(folder, class_name)
            if os.path.isdir(classdir):
                images_class = os.listdir(classdir)
                images_class.sort()
                images_class = [os.path.join(classdir,img) for img in images_class]
                indices_class = np.arange(len(images), len(images) + len(images_class))
                images.extend(images_class)
                labels.extend(len(images_class) * [label])
        self.images = np.array(images, dtype=np.object)
        self.labels = np.array(labels, dtype=np.int32)
        self.init_classes()

    def init_from_list(self, filename):
        with open(filename, 'r') as f:
            lines = f.readlines()
        lines = [line.strip().split(' ') for line in lines]
        assert len(lines)>0, \
            'List file must be in format: "fullpath(str) label(int)"'
        images = [line[0] for line in lines]
        if len(lines[0]) > 1:
            labels = [int(line[1]) for line in lines]
        else:
            labels = [os.path.dirname(img) for img in images]
            _, labels = np.unique(labels, return_inverse=True)
        if len(lines[0]) > 2:
            minutiae_maps = [line[2].strip() for line in lines]
            self.minutiae_maps = np.array(minutiae_maps, dtype=np.object)
        self.images = np.array(images, dtype=np.object)
        self.labels = np.array(labels, dtype=np.int32)
        self.init_classes()


    def init_from_hdf5(self, filename):
        with h5py.File(filename, 'r') as f:
            self.images = np.array(f['images'])
            self.labels = np.array(f['labels'])
        self.init_classes()


    def init_classes(self):
        dict_classes = {}
        classes = []
        for i, label in enumerate(self.labels):
            if not label in dict_classes:
                dict_classes[label] = [i]
            else:
                dict_classes[label].append(i)
        for label, indices in dict_classes.items():
            classes.append(self.DataClass(str(label), indices, label))
        self.classes = np.array(classes, dtype=np.object)
        self.num_classes = len(classes)
       
    def init_crossval_folder(self, folder):
        folder = os.path.expanduser(folder)
        classes = []
        images = []
        labels = []
        k_folds_classes = []
        splits = os.listdir(folder)
        splits.sort()
        for splitdir in splits:
            splitdir = os.path.join(folder, splitdir)
            fold_classes = []
            class_names = os.listdir(splitdir)
            class_names.sort()
            for label, class_name in enumerate(class_names):
                classdir = os.path.join(splitdir, class_name)
                if os.path.isdir(classdir):
                    images_class = os.listdir(classdir)
                    images_class.sort()
                    images_class = [os.path.join(classdir,img) for img in images_class]
                    indices_class = np.arange(len(images), len(images) + len(images_class))
                    images.extend(images_class)
                    labels.extend(len(images_class) * [label])
                    new_class = self.DataClass(class_name, indices_class, label)
                    classes.append(new_class)
                    fold_classes.append(new_class)
            k_folds_classes.append(fold_classes)
                    
        self.classes = np.array(classes, dtype=np.object)
        self.images = np.array(images, dtype=np.object)
        self.labels = np.array(labels, dtype=np.int32)
        self.num_classes = len(classes)
        self.k_folds_classes = k_folds_classes
        
    def import_features(self, listfile, features):
        assert self.images.shape[0] == features.shape[0]
        with open(listfile, 'r') as f:
            images = f.readlines()
        img2idx = {}
        for i, image in enumerate(images):
            img2idx[os.path.abspath(image.strip())] = i
        self.features = np.ndarray((features.shape[0], features.shape[1]), dtype=np.float)
        for i in range(self.images.shape[0]):
            self.features[i] = features[img2idx[os.path.abspath(self.images[i])]]
        return self.features
        

    def merge_with(self, dataset, mix_labels=True):
        images = np.concatenate([self.images, dataset.images], axis=0)
        if mix_labels:
            labels = np.concatenate([self.labels, dataset.labels], axis=0)
        else:
            _, labels1 = np.unique(self.labels, return_inverse=True)
            _, labels2 = np.unique(dataset.labels, return_inverse=True)
            labels2 = labels2 + np.max(labels1)
            labels = np.concatenate([labels1, labels2], axis=0)
        if self.features is not None and dataset.features is not None:
            features = np.concatenate([self.features, dataset.features])
    
        new_dataset = Dataset()
        new_dataset.images = images
        new_dataset.labels = labels
        new_dataset.features = features
        new_dataset.init_classes()
        
        print('built new dataset: %d images of %d classes' % (len(new_dataset.images), new_dataset.num_classes))

        return new_dataset

    def build_subset_from_classes(self, classes):

        if type(classes[0]) is not self.DataClass:
            try:
                classes = self.classes[classes]
            except:
                raise TypeError('The classes argument should be either self.DataClass or indices!')

        images = []
        labels = []
        features = []
        classes_new = []
        for i, c in enumerate(classes):
            n = len(c.indices)
            images.extend(self.images[c.indices])
            labels.extend([i] * n)
            if self.features is not None:
                features.append(self.features[c.indices,:].copy())
        subset = Dataset()
        subset.images = np.array(images, dtype=np.object)
        subset.labels = np.array(labels, dtype=np.int32)
        if self.features is not None:
            subset.features = np.concatenate(features, axis=0)
        subset.init_classes()

        print('built subset: %d images of %d classes' % (len(subset.images), subset.num_classes))
        return subset

    def separate_by_ratio(self, ratio, random_sort=True):
        num_classes = int(len(self.classes) * ratio)
        if random_sort:
            indices = np.random.permutation(len(self.classes))
        else:
            indices = np.arange(len(self.classes))
        indices1, indices2 = (indices[:num_classes], indices[num_classes:])
        classes1 = self.classes[indices1]
        classes2 = self.classes[indices2]
        return self.build_subset_from_classes(classes1), self.build_subset_from_classes(classes2)

    def split_k_folds(self, k, random_sort=True):
        self.k_folds_classes = []
        length = int(np.ceil(float(len(self.classes)) / k))
        if random_sort:
            indices = np.random.permutation(len(self.classes))
        else:
            indices = np.arange(len(self.classes))
        for i in range(k):
            start_ = i * length
            end_ = min(len(self.classes), (i+1) * length)
            self.k_folds_classes.append(self.classes[indices[start_:end_]])

    def get_fold(self, fold):
        k = len(self.k_folds_classes)
        assert fold <= k
        # Concatenate the classes in difference folds for trainset
        trainset_classes = [c for i in range(k) if i!=fold for c in self.k_folds_classes[i]]
        testset_classes = self.k_folds_classes[fold]
        trainset = self.build_subset_from_classes(trainset_classes)
        testset = self.build_subset_from_classes(testset_classes)
        return trainset, testset

    # Data Loading
    def init_index_queue(self, batch_format):
        if self.index_queue is None:
            self.index_queue = Queue()
        
        if batch_format == 'random_samples':
            size = self.images.shape[0]
            index_queue = np.random.permutation(size)[:,None]
        else:
            raise ValueError('IndexQueue: Unknown batch_format: {}!'.format(batch_format))
        for idx in list(index_queue):
            self.index_queue.put(idx)


    def get_batch(self, batch_size, batch_format):
        ''' Get the indices from index queue and fetch the data with indices.'''
        indices_batch = []
        
        if batch_format =='random_samples':
            while len(indices_batch) < batch_size:
                indices_batch.extend(self.index_queue.get(block=True, timeout=30)) 
            assert len(indices_batch) == batch_size
        elif batch_format == 'random_pairs':
            assert batch_size%2 == 0
            classes = np.random.permutation(self.classes)[:batch_size//2]
            indices_batch = np.concatenate([c.random_pair() for c in classes], axis=0)
        elif batch_format.startswith('random_classes'):
            try:
                _, num_classes = batch_format.split(':')
                #num_classes = int(num_classes)
                num_classes = 100 # HACK
            except:
                print('Use batch_format in such a format: random_classes: $NUM_CLASSES')
            #assert batch_size % num_classes == 0
            #num_samples_per_class = batch_size // num_classes
            num_samples_per_class = 20 # HACK
            idx_classes = np.random.permutation(self.num_classes)[:num_classes]
            indices_batch = []
            for data_class in self.classes[idx_classes]:
                indices_batch.extend(data_class.random_samples(num_samples_per_class))
            # HACK
            indices_batch = random.sample(indices_batch, batch_size)
        else:
            raise ValueError('get_batch: Unknown batch_format: {}!'.format(batch_format))

        image_batch = self.images[indices_batch]
        label_batch = self.labels[indices_batch]
        minutiae_batch = self.minutiae_maps[indices_batch]
        batch = {
            'images': image_batch,
            'labels': label_batch,
            'minutiae_labels': minutiae_batch,
        }
        return batch

    # Multithreading preprocessing images
    def start_index_queue(self, batch_format):
        if not batch_format in ['random_samples']:
            return
        self.index_queue = Queue()
        def index_queue_worker():
            while True:
                if self.index_queue.empty():
                    self.init_index_queue(batch_format)
                time.sleep(0.01)
        self.index_worker = Process(target=index_queue_worker)
        self.index_worker.daemon = True
        self.index_worker.start()

    def start_batch_queue(self, batch_size, batch_format, proc_func=None, mproc_func=None, maxsize=1, num_threads=4):

        if self.index_queue is None:
            self.start_index_queue(batch_format)

        self.batch_queue = Queue(maxsize=maxsize)
        def batch_queue_worker(seed):
            np.random.seed(seed)
            while True:
                batch = self.get_batch(batch_size, batch_format)
                if proc_func is not None:
                    batch['image_paths'] = batch['images']
                    batch['minutiae_map_paths'] = batch['minutiae_labels']
                    images, mmaps = proc_func(batch['image_paths'], batch['minutiae_map_paths'])
                    batch['images'] = images
                    batch['minutiae_labels'] = mmaps
                self.batch_queue.put(batch)

        self.batch_workers = []
        for i in range(num_threads):
            worker = Process(target=batch_queue_worker, args=(i,))
            worker.daemon = True
            worker.start()
            self.batch_workers.append(worker)
    
    def pop_batch_queue(self, timeout=60):
        return self.batch_queue.get(block=True, timeout=timeout)
      
