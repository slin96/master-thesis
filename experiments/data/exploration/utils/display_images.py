import numpy as np


def gallery(pil_images, ncols=3):
    array = make_array(pil_images)

    nindex, height, width, intensity = array.shape
    nrows = nindex // ncols
    assert nindex == nrows * ncols
    # want result.shape = (height*nrows, width*ncols, intensity)
    result = (array.reshape(nrows, ncols, height, width, intensity)
              .swapaxes(1, 2)
              .reshape(height * nrows, width * ncols, intensity))
    return result


def make_array(pil_images):
    images = list(map(lambda x: np.asarray(x.convert('RGB')), pil_images))
    return np.array(images)
