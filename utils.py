import numpy as np


# Particle binning function in 2D
# copied from https://stackoverflow.com/questions/61325586/fast-way-to-bin-a-2d-array-in-python
def bin2d(orig_array, bin_width):
    m_bins = orig_array.shape[0]//bin_width
    n_bins = orig_array.shape[1]//bin_width
    return orig_array.reshape(m_bins, bin_width, n_bins, bin_width).sum(3).sum(1)


# Color map function below shared by Lars Larson
def color_change_white(img1, img2, scale=1.0, type='conc'):
    img1 = img1 ** scale
    img2 = img2 ** scale
    img1[img1 > 1] = 1
    img1[img1 < 0] = 0
    img2[img2 > 1] = 1
    img2[img2 < 0] = 0
    comb_img = np.zeros((*img1.shape, 3))

    # Red
    comb_img[:, :, 0] = 1 - img1
    # Blue
    comb_img[:, :, 2] = 1 - img2
    # Green
    if type == 'conc':
        comb_img[:, :, 1] = 1 - (img1 + img2)
    elif type == 'rxn':
        comb_img[:, :, 1] = 1 - (img1 * img2)
    else:
        print('invalid type - options are conc or rxn.')
    comb_img[:, :, 1] = np.clip(comb_img[:, :, 1], 0, 1)  # Clip the green channel

    comb_img = np.real(comb_img)
    comb_img[comb_img > 1] = 1

    return comb_img

