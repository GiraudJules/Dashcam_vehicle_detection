import cv2
import matplotlib.image as mpimg
import numpy as np
from skimage.feature import hog


def bin_spatial(image, size=(32, 32)):
    """Compute a feature vector by resizing the image and flattening the color channels.

    Args:
        - image (numpy.ndarray): a numpy array representing an image.
        - size (tuple, optional): a tuple representing the size to resize the image to (default: (32, 32)).

    Returns:
        - a one-dimensional numpy array containing the flattened color channels of the resized image.
    """
    color1 = cv2.resize(image[:, :, 0], size).ravel()
    color2 = cv2.resize(image[:, :, 1], size).ravel()
    color3 = cv2.resize(image[:, :, 2], size).ravel()
    return np.hstack((color1, color2, color3))


def color_hist(image, bins_nb=32, bins_range=(0, 256)):
    """Compute the color histogram for an image.

    Args:
        - image: a numpy array representing an image.
        - bins_nb (int, optional): the number of bins to use for the histogram (default: 32).
        - bins_range (tuple, optional): the range of values to use for the bins (default: (0, 256)).

    Returns:
        - hist_features: a one-dimensional numpy array containing the histogram of the image
    """
    histo_channel1 = np.histogram(image[:, :, 0], bins=bins_nb, range=bins_range)
    histo_channel2 = np.histogram(image[:, :, 1], bins=bins_nb, range=bins_range)
    histo_channel3 = np.histogram(image[:, :, 2], bins=bins_nb, range=bins_range)
    hist_features = np.concatenate((histo_channel1[0], histo_channel2[0], histo_channel3[0]))
    return hist_features

def convert_RGB_color(image, color_space='RGB'):
    """
    Convert an image to the specified color space.

    Args:
        image (numpy.ndarray): Input RGB image.
        color_space (str): Desired color space ('HSV', 'LUV', 'HLS', 'YUV', 'YCrCb', or 'RGB').

    Returns:
        numpy.ndarray: Image converted to the specified color space.
    """
    conversions = {
        'HSV': cv2.COLOR_RGB2HSV,
        'LUV': cv2.COLOR_RGB2LUV,
        'HLS': cv2.COLOR_RGB2HLS,
        'YUV': cv2.COLOR_RGB2YUV,
        'YCrCb': cv2.COLOR_RGB2YCrCb
    }
    return cv2.cvtColor(image, conversions.get(color_space, cv2.COLOR_RGB2BGR))


def get_hog_features(image, orientations, pixels_per_cell, cell_per_block, visualize=False, feature_vector=True):
    """Compute the Histogram of Oriented Gradients (HOG) features for an image.

    Args:
        - image: a numpy array representing an image.
        - orientations (int): the number of orientation bins.
        - pixels_per_cell (int, int): Size (in pixels) of a cell.
        - cell_per_block (int, int): Number of cells in each block.
        - visualize (bool, optional): a boolean indicating whether to generate a visualization of the HOG features (default: False).
        - feature_vector (bool, optional): a boolean indicating whether to flatten the HOG features into a one-dimensional array (default: True).

    Returns:
        - visualize=True, return the HOG features and a visualization of the HOG features.
        - visualize=False, return the HOG features as a one-dimensional numpy array.
    """
    if visualize == True:
        features, hog_image = hog(image, orientations=orientations,
                                  pixels_per_cell=(pixels_per_cell, pixels_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=True,
                                  visualize=visualize,
                                  feature_vector=feature_vector)
        return features, hog_image

    else:
        features = hog(image, orientations=orientations,
                       pixels_per_cell=(pixels_per_cell, pixels_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       transform_sqrt=True,
                       visualize=visualize,
                       feature_vector=feature_vector)
        return features


def extract_features(imgs,
                     color_space='RGB',
                     spatial_size=(32, 32),
                     hist_bins=32,
                     orientations=9,
                     pix_per_cell=8,
                     cell_per_block=2,
                     hog_channel=0,
                     spatial_feat=True,
                     hist_feat=True,
                     hog_feat=True):
    """
    Extract hog features from a list of images.

    Args:
        imgs (list): List of image file paths.
        color_space (str): Desired color space for feature extraction ('RGB', 'HSV', 'LUV', 'HLS', 'YUV', 'YCrCb').
        spatial_size (tuple): Spatial binning dimensions.
        hist_bins (int): Number of histogram bins.
        orientations (int): Number of HOG orientations.
        pix_per_cell (int): HOG pixels per cell.
        cell_per_block (int): HOG cells per block.
        hog_channel (int or str): Which channel to use for HOG features ('0', '1', '2', or 'ALL').
        spatial_feat (bool): Flag to indicate if spatial features should be calculated.
        hist_feat (bool): Flag to indicate if histogram features should be calculated.
        hog_feat (bool): Flag to indicate if HOG features should be calculated.

    Returns:
        list: List of feature vectors for each image.
    """
    vector_features = []
    # Iterate through the list of images
    for file in imgs:
        image = mpimg.imread(file)
        feature_image = convert_RGB_color(image, color_space=color_space)

        file_features = []

        if spatial_feat:
            file_features.append(bin_spatial(feature_image, size=spatial_size))

        if hist_feat:
            file_features.append(color_hist(feature_image, bins_nb=hist_bins))

        if hog_feat:
            if hog_channel == 'ALL':
                hog_features = [get_hog_features(feature_image[:, :, i], orientations,
                                                 pix_per_cell, cell_per_block,
                                                 visualize=False, feature_vector=True) for i in range(feature_image.shape[2])]
                hog_features = np.ravel(hog_features)
            else:
                hog_features = get_hog_features(feature_image[:, :, hog_channel], orientations,
                                                pix_per_cell, cell_per_block,
                                                visualize=False, feature_vector=True)
            file_features.append(hog_features)

        vector_features.append(np.hstack(file_features))

    return vector_features
