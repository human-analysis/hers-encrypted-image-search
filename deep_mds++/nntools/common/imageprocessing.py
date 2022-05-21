"""Functions for image processing
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
import math
import random
import numpy as np
from scipy import misc
from scipy.ndimage.interpolation import rotate
import scipy.ndimage as nd
#from skimage import exposure
import cv2

# Calulate the shape for creating new array given (w,h)
def get_new_shape(images, size):
    w, h = tuple(size)
    shape = list(images.shape)
    shape[1] = h
    shape[2] = w
    shape = tuple(shape)
    return shape

def random_crop(images, size):
    n, _h, _w = images.shape[:3]
    w, h = tuple(size)
    shape_new = get_new_shape(images, size)
    assert (_h>=h and _w>=w)

    images_new = np.ndarray(shape_new, dtype=images.dtype)

    y = np.random.randint(low=0, high=_h-h+1, size=(n))
    x = np.random.randint(low=0, high=_w-w+1, size=(n))

    for i in range(n):
        images_new[i] = images[i, y[i]:y[i]+h, x[i]:x[i]+w]

    return images_new, y, x

def random_center_crop(images):
    n, _h, _w = images.shape[:3]
    final_images = []

    random_sizes = np.random.randint(low=85, high=99, size=(n))

    for i in range(n):
        rand_size = int(random_sizes[i] * 0.01 * 448)

        y = int(round(0.5 * (_h-rand_size)))
        x = int(round(0.5 * (_w-rand_size)))
        
        curr_img = images[i]
        new_img = curr_img[y:y+rand_size, x:x+rand_size]
        new_img = misc.imresize(new_img, (448, 448))
        final_images.append(new_img)
    final_images = np.array(final_images)
    return final_images, random_sizes

def random_minutiae_center_crop(maps, random_sizes):

    n, _h, _w = maps.shape[:3]
    
    final_maps = []
    for i in range(n):

        rand_size = int(random_sizes[i] * .01 * 128)

        y = int(round(0.5 * (_h - rand_size)))
        x = int(round(0.5 * (_w - rand_size)))

        curr_map = maps[i]
        new_map = curr_map[y:y+rand_size, x:x+rand_size, :]
        c1 = misc.imresize(new_map[:, :, 0:3], (128,128))
        c2 = misc.imresize(new_map[:, :, 3:6], (128,128))
        new_map = np.dstack([c1, c2])
        final_maps.append(new_map)
    final_maps = np.asarray(final_maps)
    return final_maps

def minutiae_random_crop(maps, y, x, size):
    n, _h, _w = maps.shape[:3]
    w, h = tuple(size)
    shape_new = get_new_shape(maps, size)
    assert (_h>=h and _w>=w)

    maps_new = np.ndarray(shape_new, dtype=maps.dtype)

    ratio = 138.0 / 480.0

    for i in range(n):
      cropped_map = maps[i, int(y[i] * ratio):int(y[i]*ratio)+h, int(x[i]*ratio):int(x[i]*ratio)+w]
      assert(cropped_map.shape[0] == 128 and cropped_map.shape[1] == 128 and cropped_map.shape[2] == 6)
      maps_new[i] = cropped_map

    return maps_new
    

def center_crop(images, size):
    n, _h, _w = images.shape[:3]
    w, h = tuple(size)
    assert (_h>=h and _w>=w)

    y = int(round(0.5 * (_h - h)))
    x = int(round(0.5 * (_w - w)))

    images_new = images[:, y:y+h, x:x+w]

    return images_new

def random_flip(images):
    images_new = images
    flips = np.random.rand(images_new.shape[0])>=0.5
    
    for i in range(images_new.shape[0]):
        if flips[i]:
            images_new[i] = np.fliplr(images[i])

    return images_new

def flip(images):
    images_new = images
    for i in range(images_new.shape[0]):
        images_new[i] = np.fliplr(images[i])

    return images_new

def resize(images, size):
    n, _h, _w = images.shape[:3]
    w, h = tuple(size)
    shape_new = get_new_shape(images, size)

    images_new = np.ndarray(shape_new, dtype=images.dtype)

    for i in range(n):
        images_new[i] = misc.imresize(images[i], (h,w))

    return images_new

def minutiae_resize(maps, size):
    n, _h, _w = maps.shape[:3]
    w, h = tuple(size)
    shape_new = get_new_shape(maps, size)

    maps_new = np.ndarray(shape_new, dtype=maps.dtype)

    for i in range(n):
        mmap = maps[i]
        c1 = misc.imresize(mmap[:, :, 0:3], (h,w))
        c2 = misc.imresize(mmap[:, :, 3:6], (h,w))
        c3 = misc.imresize(mmap[:, :, 6:9], (h,w))
        c4 = misc.imresize(mmap[:, :, 9:12], (h,w))
        stacked = np.dstack([c1, c2, c3, c4])
        maps_new[i] = stacked
        
    return maps_new

def standardize_images(images, standard):
    if standard=='mean_scale':
        mean = 127.5
        std = 128.0
    elif standard=='scale':
        mean = 0.0
        std = 255.0
    images_new = images.astype(np.float32)
    images_new = (images_new - mean) / std
    return images_new

def standardize_maps(maps):
    maps_new = maps.astype(np.float32)
    maps_new = maps_new / 255.0
    return maps_new

def random_downsample(images, min_ratio):
    n, _h, _w = images.shape[:3]
    images_new = images
    ratios = min_ratio + (1-min_ratio) * np.random.rand(images_new.shape[0])

    for i in range(images_new.shape[0]):
        w = int(round(ratios[i] * _w))
        h = int(round(ratios[i] * _h))
        images_new[i,:h,:w] = misc.imresize(images[i], (h,w))
        images_new[i] = misc.imresize(images_new[i,:h,:w], (_h,_w))
        
    return images_new

def pad_image(img_data, output_width, output_height):
    height, width = img_data.shape
    output_img = np.ones((output_height, output_width), dtype=np.int32) * 255
    margin_h = (output_height - height) // 2
    margin_w = (output_width - width) // 2
    output_img[margin_h:margin_h+height, margin_w:margin_w+width] = img_data
    return output_img

def pad_map(map_data, output_width, output_height):
    height, width, channels = map_data.shape
    output_map = np.ones((output_height, output_width, channels), dtype=np.float32)
    margin_h = (output_height - height) // 2
    margin_w = (output_width - width) // 2
    output_map[margin_h:margin_h+height, margin_w:margin_w+width, :] = map_data
    return output_map

def random_brightness(images, max_delta):
    # randomly adjust the brightness of the image by -max delta to max_delta
    images_new = images
    for i in range(images_new.shape[0]):
        delta = random.uniform(-1.0*max_delta, max_delta)
        images_new[i] = np.clip(images[i] + delta, 0, 255)
    return images_new

def random_rotate(images, degrees):
    # randomly rotate from 0 to degrees and from 0 to negative degrees
    images_new = images
    ret_degs = []
    for i in range(images_new.shape[0]):
      deg = random.uniform(-1.0*degrees, degrees)
      if deg < 0:
        deg = 360 + deg
      ret_degs.append(deg)
      images_new[i] = rotate(images[i], deg, reshape=False, cval=255.0)
    return images_new, ret_degs

def rotate_minutiae_maps(maps, degrees, size):
    n, _h, _w = maps.shape[:3]
    w, h = tuple(size)
    shape_new = get_new_shape(maps, size)

    maps_new = np.ndarray(shape_new, dtype=maps.dtype)

    for i in range(n):
        mmap = maps[i]
        degs = degrees[i]
        c1 = rotate(mmap[:, :, 0:3], degs, reshape=False, cval=0.0)
        c2 = rotate(mmap[:, :, 3:6], degs, reshape=False, cval=0.0)
        stacked = np.dstack([c1, c2])
        maps_new[i] = stacked
    return maps_new

def translate_minutiae_maps(maps, translations, size):
    n, _h, _w = maps.shape[:3]
    w, h = tuple(size)
    shape_new = get_new_shape(maps, size)

    maps_new = np.ndarray(shape_new, dtype=maps.dtype)

    ratio = 138.0 / 480.0

    for i in range(n):
        mmap = maps[i]
        x_units, y_units = translations[i]
        x_units = int(x_units * ratio)
        y_units = int(y_units * ratio)

        c1 = nd.interpolation.affine_transform(np.squeeze(mmap[:, :, 0:1]),((np.cos(0), np.sin(0)), (-np.sin(0), np.cos(0))), offset=(x_units,y_units),order=3, cval=0.0)
        c2 = nd.interpolation.affine_transform(np.squeeze(mmap[:, :, 1:2]),((np.cos(0), np.sin(0)), (-np.sin(0), np.cos(0))), offset=(x_units,y_units),order=3, cval=0.0)
        c3 = nd.interpolation.affine_transform(np.squeeze(mmap[:, :, 2:3]),((np.cos(0), np.sin(0)), (-np.sin(0), np.cos(0))), offset=(x_units,y_units),order=3, cval=0.0)
        c4 = nd.interpolation.affine_transform(np.squeeze(mmap[:, :, 3:4]),((np.cos(0), np.sin(0)), (-np.sin(0), np.cos(0))), offset=(x_units,y_units),order=3, cval=0.0)
        c5 = nd.interpolation.affine_transform(np.squeeze(mmap[:, :, 4:5]),((np.cos(0), np.sin(0)), (-np.sin(0), np.cos(0))), offset=(x_units,y_units),order=3, cval=0.0)
        c6 = nd.interpolation.affine_transform(np.squeeze(mmap[:, :, 5:6]),((np.cos(0), np.sin(0)), (-np.sin(0), np.cos(0))), offset=(x_units,y_units),order=3, cval=0.0)
        stacked = np.dstack([c1, c2, c3, c4, c5, c6])
        maps_new[i] = stacked
    return maps_new

def random_contrast(images, placeholder):
    print("PLEASE UNCOMMENT ME!!")
    return images
    """
    # e.g. 99th percentile is 99.0
    images_new = images
    for i in range(images_new.shape[0]):
      frac = random.uniform(-1.0,  1.0)
      if frac > 0.0:
          out = (0, 255)
      else:
          scale = random.uniform(75.0, 112.5)
          lb_out = scale
          ub_out = 255 - scale
          out = (lb_out, ub_out)
      lb_in = random.uniform(0.0, 2.0)
      ub_in = 100 - lb_in
      v_min, v_max = np.percentile(images[i], (lb_in, ub_in))
      random_contrast = exposure.rescale_intensity(images[i], in_range=(v_min, v_max), out_range=out)
      images_new[i] = random_contrast
    return images_new
    """

def random_translate(images, units):
    units_x, units_y = units
    images_new = images
    translations = []
    for i in range(images_new.shape[0]):
      units_x = random.randint(0, units_x)
      units_y = random.randint(0, units_y)
      flip_x = random.randint(-1, 1)
      flip_y = random.randint(-1, 1)
      x_units = int(flip_x * units_x)
      y_units = int(flip_y * units_y)

      translated = nd.interpolation.affine_transform(images[i],((np.cos(0), np.sin(0)), (-np.sin(0), np.cos(0))), offset=(x_units,y_units),order=3, cval=255.0)
      images_new[i] = translated
      translations.append((x_units, y_units))
    return images_new, translations


def random_obfuscations(images, block_size, fraction):
    images_new = images
    all_indices = []
    for i in range(images_new.shape[0]):
        num = fraction * 100.0
        frac = random.uniform(-1.0*num, num)
        if frac < 0:
            frac = 0
        frac = frac / 100.0
        obfuscated, indices = random_obfuscation(images[i], block_size, frac)
        images_new[i] = obfuscated
        all_indices.append(indices)
    return images_new, all_indices

def random_obfuscation_mmaps(mmaps, block_size, indices):
    mmaps_new = mmaps
    for i in range(mmaps_new.shape[0]):
        obfuscated = random_obfuscation_mmap(mmaps[i], block_size, indices[i])
        mmaps_new[i] = obfuscated
    return mmaps_new

def random_obfuscation(img, block_size, fraction):
    height, width = img.shape

    if height % block_size != 0 or width % block_size != 0:
        print("block size must evely divide into img width {} and height {}".format(width, height))

    num_blocks = (height // block_size) * (width // block_size) 
    cells = list(range(num_blocks))


    cells = random.sample(cells, int(fraction * num_blocks))
    indices = [0] * num_blocks
    for e in cells: indices[e] = 1

    i = 0
    for r in range(0, height, block_size):
        for c in range(0, width, block_size):
            if indices[i] == 1:
                # set this block to zeros
                img[r:r+block_size, c:c+block_size] = np.ones((block_size, block_size)) * 255
            i += 1

    return img, indices

def random_obfuscation_mmap(mmap, block_size, indices):
    height, width, channels = mmap.shape
    i = 0
    for r in range(0, height, block_size):
        for c in range(0, width, block_size):
            if indices[i] == 1:
                mmap[r:r+block_size, c:c+block_size, :] = np.zeros((block_size, block_size, channels))
            i += 1

    return mmap

def blur(image):
    image = cv2.GaussianBlur(image, ksize=(5, 5), sigmaX=5)
    return image


def preprocess_test(images, config, is_training=False):
    # Load images first if they are file paths
    if type(images[0]) == str:
        image_paths = images
        images = []
        assert (config.channels==1 or config.channels==3)
        mode = 'RGB' if config.channels==3 else 'I'
        for i, image_path in enumerate(image_paths):
            #img = misc.imread(image_path, mode=mode)
            img = cv2.imread(image_path, 0)
            #img = blur(img)
            #img = cv2.equalizeHist(img)
            if (img.shape[0] != 512 or img.shape[1] != 512):
              img = pad_image(img, 512, 512)
            images.append(img)
        images = np.stack(images, axis=0)

    # Process images
    f = {
        'resize': resize,
        'random_crop': random_crop,
        'center_crop': center_crop,
        'random_flip': random_flip,
        'random_rotate': random_rotate,
        'random_brightness' : random_brightness,
        'random_translate' : random_translate,
        'standardize': standardize_images,
        'random_downsample': random_downsample,
        'random_contrast' : random_contrast
    }
    proc_funcs = config.preprocess_train if is_training else config.preprocess_test

    for proc in proc_funcs:
        proc_name, proc_args = proc[0], proc[1:]
        if proc_name == 'random_crop':
          images, y, x = f[proc_name](images, *proc_args)
        else:
          images = f[proc_name](images, *proc_args)

    if len(images.shape) == 3:
        images = images[:,:,:,None]
    return images

def preprocess(images, maps_paths,config, is_training=False):
    # Load images first if they are file paths
    if type(images[0]) == str:
        image_paths = images
        images = []
        maps = []
        assert (config.channels==1 or config.channels==3)
        mode = 'RGB' if config.channels==3 else 'I'
        for i, image_path in enumerate(image_paths):
            #img = misc.imread(image_path, mode=mode)
            img = cv2.imread(image_path, 0)
            mmap = np.load(maps_paths[i])
            if (img.shape[0] != 512 or img.shape[1] != 512) and is_training:
              img = pad_image(img, 512, 512)
            #img = blur(img)
            images.append(img)
            maps.append(mmap)
        images = np.stack(images, axis=0)
        maps = np.stack(maps, axis=0)


    # Process images
    f = {
        'resize': resize,
        'random_crop': random_crop,
        'center_crop': center_crop,
        'random_flip': random_flip,
        'random_rotate': random_rotate,
        'random_brightness' : random_brightness,
        'random_translate' : random_translate,
        'standardize': standardize_images,
        'random_downsample': random_downsample,
        'random_contrast' : random_contrast,
        'random_obfuscation' : random_obfuscations,
        'random_downsample' : random_center_crop
    }
    proc_funcs = config.preprocess_train if is_training else config.preprocess_test

    for proc in proc_funcs:
        proc_name, proc_args = proc[0], proc[1:]
        if proc_name == 'random_crop':
            images, y, x = f[proc_name](images, *proc_args)
        elif proc_name == 'random_translate':
            images, translations = f[proc_name](images, *proc_args)
        elif proc_name == 'random_rotate':
            images, degrees = f[proc_name](images, *proc_args)
        elif proc_name == 'random_obfuscation':
            images, indices = f[proc_name](images, 28, .25)
        elif proc_name == 'random_downsample':
            images, random_sizes = f[proc_name](images)
        else:
            images = f[proc_name](images, *proc_args)
    
    maps = rotate_minutiae_maps(maps, degrees, (138, 138))
    maps = translate_minutiae_maps(maps, translations, (138, 138))
    maps = minutiae_random_crop(maps, y, x, (128, 128))
    #maps = random_minutiae_center_crop(maps, random_sizes)
    #maps = standardize_maps(maps)
    #maps = random_obfuscation_mmaps(maps, 8, indices) 

    if len(images.shape) == 3:
        images = images[:,:,:,None]
    if len(maps.shape) == 3:
        maps = maps[:, :, :, None]
    return images, maps
        

