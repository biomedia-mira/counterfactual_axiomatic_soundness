import numpy as np
import cv2
from scipy.fft import dct
from skimage


zigzag = ((0, 0), (0, 1), (1, 0), (2, 0), (1, 1), (0, 2), ())


def to_YCbCr(r, g, b):
    y = .299 * r + .587 * g + .114 * b
    cb = 128 - .168736 * r - .331264 * g + .5 * b
    cr = 128 + .5 * r - .418688 * g - .081312 * b
    return y, cb, cr


def to_RGB(y, cb, cr):
    r = y + 1.402 * (cr - 128)
    g = y - .344136 * (cb - 128) - .714136 * (cr - 128)
    b = y + 1.772 + (cb - 128)
    return r, g, b


def padding(image, block_size):
    pass


def array_to_blocks(array, nrows, ncols):
    h, w = array.shape
    return (array.reshape(h // nrows, nrows, -1, ncols).swapaxes(1, 2).reshape(-1, nrows, ncols))


def encoder(image, block_size):
    image = padding(image)
    y, cb, cr = to_YCbCr(*image)
    nrows, ncols = 4, 4

    y_blocks = array_to_blocks(y, nrows, ncols)
    cb_blocks = array_to_blocks(cb, nrows, ncols)
    cr_blocks = array_to_blocks(cb, nrows, ncols)

    for block in blocks:
        dct_blocks = dct(block)


img = np.round(np.random.random(size=(28, 28, 3)) * 255).astype(int)

encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
result, encimg = cv2.imencode('.jpg', img, encode_param)
