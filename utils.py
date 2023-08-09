import glob
import os

import matplotlib.patches as patches
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from skimage.io import imread


def read_images(directory, image_array):
    """
    Read images from a given directory and append to a list.

    Args:
        directory (str): Path to the directory containing images.
        image_array (list): List to which image paths will be appended.
    """
    images = glob.iglob(directory + '/*.png', recursive=True)
    for image in images:
        image_array.append(image)


def read_frame(df_annotation, frame):
    """
    Fetch an image corresponding to a specific frame ID.

    Args:
        df_annotation (pd.DataFrame): A dataframe containing annotations, indexed by frame ID.
        frame (int): The specific frame ID to fetch the image for.

    Returns:
        np.array: The image corresponding to the specified frame ID.
    """
    file_path = df_annotation[df_annotation.index == frame]['frame_id'].values[0]
    return imread(file_path)


def annotations_for_frame(df_annotation, frame):
    """
    Retrieve bounding box annotations for a given frame.

    Args:
        df_annotation (pd.DataFrame): A dataframe containing annotations, indexed by frame ID.
        frame (int): The frame ID for which bounding box annotations are required.

    Returns:
        list: A list of bounding box annotations. Each bounding box is represented as [x, y, dx, dy].
    """
    assert frame in df_annotation.index
    bbs = df_annotation[df_annotation.index == frame].bounding_boxes.values[0]

    if pd.isna(bbs): # some frames contain no vehicles
        return []

    bbs = list(map(lambda x : int(x), bbs.split(' ')))
    return np.array_split(bbs, len(bbs) / 4)


def show_annotation(df_annotation, frame):
    """
    Display a frame image with its associated bounding box annotations.

    Args:
        df_annotation (pd.DataFrame): A dataframe containing annotations, indexed by frame ID.
        frame (int): The frame ID of the image to be displayed with annotations.

    Returns:
        None. The function displays an image using matplotlib.
    """
    img = read_frame(df_annotation, frame)
    bbs = annotations_for_frame(df_annotation, frame)

    fig, ax = plt.subplots(figsize=(10, 8))

    for x, y, dx, dy in bbs:

        rect = patches.Rectangle((x, y), dx, dy, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    ax.imshow(img)
    ax.set_title('Annotations for frame {}.'.format(frame))


def bounding_boxes_to_mask(bounding_boxes, H, W):
    """Convert set of bounding boxes to a binary mask."""
    mask = np.zeros((H, W))
    for x, y, dx, dy in bounding_boxes:
        mask[y:y+dy, x:x+dx] = 1

    return mask


def run_length_encoding(mask):
    """Produce run length encoding for a given binary mask."""
    # find mask non-zeros in flattened representation
    non_zeros = np.nonzero(mask.flatten())[0]

    if len(non_zeros) == 0:
        return ''

    padded = np.pad(non_zeros, pad_width=1, mode='edge')

    # find start and end points of non-zeros runs
    limits = (padded[1:] - padded[:-1]) != 1
    starts = non_zeros[limits[:-1]]
    ends = non_zeros[limits[1:]]
    lengths = ends - starts + 1

    return ' '.join(['%d %d' % (s, l) for s, l in zip(starts, lengths)])
