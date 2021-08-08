
# Code taken from: https://github.com/IFL-CAMP/dense_adversarial_generation_pytorch and slightly modified

import os
import torch
import numpy as np
import scipy.misc as smp
import scipy.ndimage
from random import randint

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def make_one_hot(labels, num_classes, device):
    '''
    Converts an integer label torch.autograd.Variable to a one-hot Variable.
    Parameters
    ----------
        labels : torch.LongTensor
        N x 1 x H x W, where N is batch size.
        Each value is an integer representing correct classification.
        num_classes : int
        Number of classes
        device: string
        Device to place the new tensor on. Should be same as input
    Returns
    -------
        target : torch.Tensor on given device
        N x C x H x W, where C is class number. One-hot encoded.
    '''
    labels=labels.unsqueeze(1)
    #print("Labels here is",labels.shape,labels.type())
    one_hot = torch.FloatTensor(labels.size(0), num_classes, labels.size(2), labels.size(3)).zero_()
    one_hot = one_hot.to(device)
    target = one_hot.scatter_(1, labels.data, 1)
    return target

import numpy as np
from PIL import Image


_errstr = "Mode is unknown or incompatible with input array shape."


def bytescale(data, cmin=None, cmax=None, high=255, low=0):
    """
    Byte scales an array (image).
    Byte scaling means converting the input image to uint8 dtype and scaling
    the range to ``(low, high)`` (default 0-255).
    If the input image already has dtype uint8, no scaling is done.
    This function is only available if Python Imaging Library (PIL) is installed.
    Parameters
    ----------
    data : ndarray
        PIL image data array.
    cmin : scalar, optional
        Bias scaling of small values. Default is ``data.min()``.
    cmax : scalar, optional
        Bias scaling of large values. Default is ``data.max()``.
    high : scalar, optional
        Scale max value to `high`.  Default is 255.
    low : scalar, optional
        Scale min value to `low`.  Default is 0.
    Returns
    -------
    img_array : uint8 ndarray
        The byte-scaled array.
    Examples
    --------
    >>> from scipy.misc import bytescale
    >>> img = np.array([[ 91.06794177,   3.39058326,  84.4221549 ],
    ...                 [ 73.88003259,  80.91433048,   4.88878881],
    ...                 [ 51.53875334,  34.45808177,  27.5873488 ]])
    >>> bytescale(img)
    array([[255,   0, 236],
           [205, 225,   4],
           [140,  90,  70]], dtype=uint8)
    >>> bytescale(img, high=200, low=100)
    array([[200, 100, 192],
           [180, 188, 102],
           [155, 135, 128]], dtype=uint8)
    >>> bytescale(img, cmin=0, cmax=255)
    array([[91,  3, 84],
           [74, 81,  5],
           [52, 34, 28]], dtype=uint8)
    """
    if data.dtype == np.uint8:
        return data

    if high > 255:
        raise ValueError("`high` should be less than or equal to 255.")
    if low < 0:
        raise ValueError("`low` should be greater than or equal to 0.")
    if high < low:
        raise ValueError("`high` should be greater than or equal to `low`.")

    if cmin is None:
        cmin = data.min()
    if cmax is None:
        cmax = data.max()

    cscale = cmax - cmin
    if cscale < 0:
        raise ValueError("`cmax` should be larger than `cmin`.")
    elif cscale == 0:
        cscale = 1

    scale = float(high - low) / cscale
    bytedata = (data - cmin) * scale + low
    return (bytedata.clip(low, high) + 0.5).astype(np.uint8)


def toimage(arr, high=255, low=0, cmin=None, cmax=None, pal=None,
            mode=None, channel_axis=None):
    """Takes a numpy array and returns a PIL image.
    This function is only available if Python Imaging Library (PIL) is installed.
    The mode of the PIL image depends on the array shape and the `pal` and
    `mode` keywords.
    For 2-D arrays, if `pal` is a valid (N,3) byte-array giving the RGB values
    (from 0 to 255) then ``mode='P'``, otherwise ``mode='L'``, unless mode
    is given as 'F' or 'I' in which case a float and/or integer array is made.
    .. warning::
        This function uses `bytescale` under the hood to rescale images to use
        the full (0, 255) range if ``mode`` is one of ``None, 'L', 'P', 'l'``.
        It will also cast data for 2-D images to ``uint32`` for ``mode=None``
        (which is the default).
    Notes
    -----
    For 3-D arrays, the `channel_axis` argument tells which dimension of the
    array holds the channel data.
    For 3-D arrays if one of the dimensions is 3, the mode is 'RGB'
    by default or 'YCbCr' if selected.
    The numpy array must be either 2 dimensional or 3 dimensional.
    """
    data = np.asarray(arr)
    if np.iscomplexobj(data):
        raise ValueError("Cannot convert a complex-valued array.")
    shape = list(data.shape)
    valid = len(shape) == 2 or ((len(shape) == 3) and
                                ((3 in shape) or (4 in shape)))
    if not valid:
        raise ValueError("'arr' does not have a suitable array shape for "
                         "any mode.")
    if len(shape) == 2:
        shape = (shape[1], shape[0])  # columns show up first
        if mode == 'F':
            data32 = data.astype(np.float32)
            image = Image.frombytes(mode, shape, data32.tostring())
            return image
        if mode in [None, 'L', 'P']:
            bytedata = bytescale(data, high=high, low=low,
                                 cmin=cmin, cmax=cmax)
            image = Image.frombytes('L', shape, bytedata.tostring())
            if pal is not None:
                image.putpalette(np.asarray(pal, dtype=np.uint8).tostring())
                # Becomes a mode='P' automagically.
            elif mode == 'P':  # default gray-scale
                pal = (np.arange(0, 256, 1, dtype=np.uint8)[:, np.newaxis] *
                       np.ones((3,), dtype=np.uint8)[np.newaxis, :])
                image.putpalette(np.asarray(pal, dtype=np.uint8).tostring())
            return image
        if mode == '1':  # high input gives threshold for 1
            bytedata = (data > high)
            image = Image.frombytes('1', shape, bytedata.tostring())
            return image
        if cmin is None:
            cmin = np.amin(np.ravel(data))
        if cmax is None:
            cmax = np.amax(np.ravel(data))
        data = (data*1.0 - cmin)*(high - low)/(cmax - cmin) + low
        if mode == 'I':
            data32 = data.astype(np.uint32)
            image = Image.frombytes(mode, shape, data32.tostring())
        else:
            raise ValueError(_errstr)
        return image

    # if here then 3-d array with a 3 or a 4 in the shape length.
    # Check for 3 in datacube shape --- 'RGB' or 'YCbCr'
    if channel_axis is None:
        if (3 in shape):
            ca = np.flatnonzero(np.asarray(shape) == 3)[0]
        else:
            ca = np.flatnonzero(np.asarray(shape) == 4)
            if len(ca):
                ca = ca[0]
            else:
                raise ValueError("Could not find channel dimension.")
    else:
        ca = channel_axis

    numch = shape[ca]
    if numch not in [3, 4]:
        raise ValueError("Channel axis dimension is not valid.")

    bytedata = bytescale(data, high=high, low=low, cmin=cmin, cmax=cmax)
    if ca == 2:
        strdata = bytedata.tostring()
        shape = (shape[1], shape[0])
    elif ca == 1:
        strdata = np.transpose(bytedata, (0, 2, 1)).tostring()
        shape = (shape[2], shape[0])
    elif ca == 0:
        strdata = np.transpose(bytedata, (1, 2, 0)).tostring()
        shape = (shape[2], shape[1])
    if mode is None:
        if numch == 3:
            mode = 'RGB'
        else:
            mode = 'RGBA'

def generate_target(y_test, xsize=448, target_class = 0):
    my_test_original = y_test
    my_test = np.argmax(y_test[0,:,:,:], axis = 1)
    preds = toimage(my_test)
    #plt.imshow(preds, cmap='jet')
    #plt.show()
    y_target = y_test


    dilated_image = scipy.ndimage.binary_dilation(y_target[0, target_class, :, :], iterations=6).astype(y_test.dtype)

    for i in range(xsize):
        for j in range(xsize):
            y_target[0, target_class, i, j] = dilated_image[i,j]

    for i in range(xsize):
        for j in range(xsize):
            potato = np.count_nonzero(y_target[0,:,i,j])
            if (potato > 1):
                x = np.where(y_target[0, : ,i, j] > 0)
                k = x[0]
                if k[0] == target_class:
                    y_target[0,k[1],i,j] = 0.
                else:
                    y_target[0, k[0], i, j] = 0.

    my_target = np.argmax(y_target[0,:,:,:], axis = 1)
    preds = toimage(my_target)
    return y_target

def generate_target_swap(y_test, xsize=448):
    #my_test_original = y_test
    #my_test = np.argmax(y_test[0,:,:,:], axis = -1)
    #preds = smp.toimage(my_test)
    #plt.imshow(preds, cmap='jet')
    #plt.show()

    y_target = y_test

    y_target_arg = np.argmax(y_test, axis = 1)

    y_target_arg_no_back = np.where(y_target_arg>0)

    y_target_arg = y_target_arg[y_target_arg_no_back]

    classes  = np.unique(y_target_arg)

    #print(classes)

    if len(classes) > 3:

        first_class = 0

        second_class = 0

        third_class = 0

        while first_class == second_class == third_class:
            first_class = classes[randint(0, len(classes)-1)]
            f_ind = np.where(y_target_arg==first_class)
            #print(np.shape(f_ind))

            second_class = classes[randint(0, len(classes)-1)]
            s_ind = np.where(y_target_arg == second_class)

            third_class = classes[randint(0, len(classes) - 1)]
            t_ind = np.where(y_target_arg == third_class)

            summ = np.shape(f_ind)[1] + np.shape(s_ind)[1] + np.shape(t_ind)[1]

            if summ < 1000:
                first_class = 0

                second_class = 0

                third_class = 0

        for i in range(xsize):
            for j in range(xsize):
                temp = y_target[0,second_class, i,j]
                y_target[0,second_class, i,j] = y_target[0,first_class,i,j]
                y_target[0, first_class,i, j] = temp

        '''
        print('New target')
        my_target = np.argmax(y_target[0,:,:,:], axis = -1)
        my_test = np.argmax(y_test[0, :, :, :], axis=-1)
        print('potato')
        print(np.shape(my_target))
        print(np.shape(my_test))
        #my_test = np.reshape(my_test, (256, 256))
        together = np.concatenate((my_test, my_target), axis = 1)
        preds = smp.toimage(together)
        plt.imshow(preds, cmap='jet')
        plt.show()
        '''
    else:
        y_target = y_test
        print('Not enough classes to swap!')
    return y_target