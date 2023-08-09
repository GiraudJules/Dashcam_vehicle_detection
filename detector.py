import os

import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import label

import hog as h
import utils as u


def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],  xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    """Generate a list of sliding windows over an image.

    Args:
        img (np.array): The image over which windows will slide.
        x_start_stop (list, optional): The start and stop x positions for the window. Defaults to [None, None].
        y_start_stop (list, optional): The start and stop y positions for the window. Defaults to [None, None].
        xy_window (tuple, optional): Window size in (width, height) format. Defaults to (64, 64).
        xy_overlap (tuple, optional): Fraction of window overlap in (x, y) format. Defaults to (0.5, 0.5).

    Returns:
        list: List of window positions, each position is a tuple of top-left and bottom-right coordinates.
    """
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]

    # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]

    # Compute the number of pixels per step in x/y
    nx_pix_per_step = int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = int(xy_window[1]*(1 - xy_overlap[1]))

    # Compute the number of windows in x/y
    nx_buffer = int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = int(xy_window[1]*(xy_overlap[1]))
    nx_windows = int((xspan-nx_buffer)/nx_pix_per_step)
    ny_windows = int((yspan-ny_buffer)/ny_pix_per_step)

    window_list = []

    # Loop through finding x and y window positions
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            window_list.append(((startx, starty), (endx, endy)))

    return window_list


def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    """Draw bounding boxes on an image.

    Args:
        img (np.array): The image on which boxes will be drawn.
        bboxes (list): List of bounding boxes, each box is represented by top-left and bottom-right coordinates.
        color (tuple, optional): Color of the boxes in (R, G, B) format. Defaults to (0, 0, 255).
        thick (int, optional): Thickness of the box lines. Defaults to 6.

    Returns:
        np.array: Image with drawn boxes.
    """
    imcopy = np.copy(img)
    random_color = False
    # Iterate through the bounding boxes
    for bbox in bboxes:
        if color == 'random' or random_color:
            color = (np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255))
            random_color = True
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    return imcopy


def view_windows(image, window_scale, x_start_stop, y_start_stop, xy_window, xy_overlap, color_values):
    """Visualize sliding windows on an image.

    Args:
        image (np.array): The image on which windows will be visualized.
        window_scale (tuple): The scales of the windows to slide over the image.
        x_start_stop (tuple): The start and stop positions in the x-dimension to search for windows.
        y_start_stop (list): List of start and stop positions in the y-dimension for each scale.
        xy_window (tuple): Window size in (width, height) format.
        xy_overlap (tuple): Fraction of window overlap in (x, y) format.
        color_values (list): List of colors for the boxes for each scale.

    Returns:
        None: The function visualizes the sliding windows on the input image using matplotlib.
    """
    fig, ax = plt.subplots(figsize=(10,10))
    for i, scale in enumerate(window_scale):
        windows = slide_window(image, x_start_stop=x_start_stop, y_start_stop=y_start_stop[i],
                               xy_window=[int(dim*window_scale[i]) for dim in xy_window], xy_overlap=xy_overlap)
        image = draw_boxes(image, windows, color_values[i])
        print("Number of windows for scale {}: {}".format(scale, len(windows)))
    plt.imshow(image)
    plt.title("Test Image")
    plt.show()


def find_cars(img, x_start_stop, ystart, ystop, scale, model, X_scaler, orientations, pix_per_cell, cell_per_block, spatial_size, hist_bins, vis_bboxes=False):
    """Find cars in an image using a sliding window approach and a trained classifier.

    Args:
        img (np.array): The image to be searched for cars.
        x_start_stop (tuple): The start and stop positions of the search in the x-dimension.
        ystart (int): The start position of the search in the y-dimension.
        ystop (int): The stop position of the search in the y-dimension.
        scale (float): The scale of the search window.
        model (classifier): The trained classifier used to predict if a window contains a car.
        X_scaler (scaler): The scaler used to normalize the feature vectors.
        orientations (int): Number of HOG orientations.
        pix_per_cell (int): Number of pixels per cell for the HOG features.
        cell_per_block (int): Number of cells per block for the HOG features.
        spatial_size (tuple): Spatial size for the spatial binning of the color channels.
        hist_bins (int): Number of histogram bins for color histogram features.
        vis_bboxes (bool, optional): If True, visualizes all bounding boxes regardless of classifier prediction. Defaults to False.

    Returns:
        list: List of bounding boxes where cars were detected.
    """
    draw_img = np.copy(img)
    xstart = 0
    xstop = img.shape[1]
    img_tosearch = img[ystart:ystop, x_start_stop[0]:x_start_stop[1], :]
    ctrans_tosearch = h.convert_RGB_color(img_tosearch, color_space='RGB')

    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (int(imshape[1]/scale), int(imshape[0]/scale)))

    ch1 = ctrans_tosearch[:, :, 0]
    ch2 = ctrans_tosearch[:, :, 1]
    ch3 = ctrans_tosearch[:, :, 2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    hog1 = h.get_hog_features(ch1, orientations=orientations, pixels_per_cell=pix_per_cell, cell_per_block=cell_per_block, feature_vector=False)
    hog2 = h.get_hog_features(ch2, orientations=orientations, pixels_per_cell=pix_per_cell, cell_per_block=cell_per_block, feature_vector=False)
    hog3 = h.get_hog_features(ch3, orientations=orientations, pixels_per_cell=pix_per_cell, cell_per_block=cell_per_block, feature_vector=False)

    rectangles = []

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step

            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3)).reshape(1, -1)

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))

            #spatial_features = h.bin_spatial(subimg, size=spatial_size).reshape(1, -1)

            hist_features = h.color_hist(subimg, bins_nb=hist_bins).reshape(1, -1)

            test_features = X_scaler.transform(np.hstack((hist_features, hog_features)))

            test_prediction = model.predict(test_features)

            if test_prediction == 1 or vis_bboxes == True:
                xbox_left = int(xleft*scale)
                ytop_draw = int(ytop*scale)
                win_draw = int(window*scale)
                rectangles.append(((xbox_left+xstart, ytop_draw+ystart), (xbox_left+win_draw+xstart, ytop_draw+win_draw+ystart)))

    return rectangles


def apply_threshold(heatmap, threshold):
    """Apply a threshold to a heatmap, setting values below the threshold to zero.

    Args:
        heatmap (np.array): The heatmap on which the threshold is to be applied.
        threshold (int): The threshold value.

    Returns:
        np.array: The thresholded heatmap.
    """
    heatmap[heatmap <= threshold] = 0
    return heatmap


def apply_adaptive_threshold(heatmap, threshold_ratio):
    """Apply an adaptive threshold to a heatmap.

    Args:
        heatmap (np.array): The heatmap on which the threshold is to be applied.
        threshold_ratio (float): The ratio of the maximum heatmap value to be used as the threshold.

    Returns:
        np.array: The thresholded heatmap.
    """
    max_val = np.amax(heatmap)
    threshold = max_val * threshold_ratio
    heatmap[heatmap <= threshold] = 0
    return heatmap


def get_rectangles(image, window_scale, x_start_stop, y_start_stop, model, X_scaler, orientations, pix_per_cell, cell_per_block, spatial_size, hist_bins):
    """Generate a list of rectangles where cars are detected in an image.

    Args:
        image (np.array): The image on which cars are to be detected.
        window_scale (tuple): The scales of the windows to slide over the image.
        x_start_stop (tuple): The start and stop positions in the x-dimension to search for windows.
        y_start_stop (list): List of start and stop positions in the y-dimension for each scale.
        model (classifier): The trained classifier used to detect cars.
        X_scaler (scaler): The scaler used to normalize the feature vectors.
        orientations (int): Number of HOG orientations.
        pix_per_cell (int): Number of pixels per cell for the HOG features.
        cell_per_block (int): Number of cells per block for the HOG features.
        spatial_size (tuple): Spatial size for the spatial binning of the color channels.
        hist_bins (int): Number of histogram bins for color histogram features.

    Returns:
        list: List of bounding boxes where cars were detected.
    """
    out_rectangles = []
    for i, scale in enumerate(window_scale):
        rectangles = find_cars(image, x_start_stop, y_start_stop[i][0], y_start_stop[i][1], scale, model, X_scaler, orientations, pix_per_cell, cell_per_block, spatial_size, hist_bins)
        if len(rectangles) > 0:
            out_rectangles.append(rectangles)
    out_rectangles = [item for sublist in out_rectangles for item in sublist]
    return out_rectangles


def add_heat(heatmap, bbox_list):
    """Increase the intensity of pixels inside each bounding box in the heatmap.

    Args:
        heatmap (np.array): The heatmap to which heat will be added.
        bbox_list (list): List of bounding boxes.

    Returns:
        np.array: Updated heatmap with added heat.
    """
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap


def draw_labeled_bboxes(img, labels):
    """Draw bounding boxes on an image based on labeled regions.

    Args:
        img (np.array): The image on which bounding boxes are to be drawn.
        labels (tuple): Labeled regions in the format (labeled_array, number_of_labels).

    Returns:
        list: List of bounding boxes drawn.
        np.array: Image with bounding boxes drawn.
    """
    img_copy = np.copy(img)
    result_rectangles = []

    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()

        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        area = (bbox[1][1] - bbox[0][1]) * (bbox[1][0] - bbox[0][0])

        if area > 40 * 40:
            x, y = bbox[0][0], bbox[0][1]
            w, h = bbox[1][0] - bbox[0][0], bbox[1][1] - bbox[0][1]
            result_rectangles.append([x, y, w, h])

            # Draw the box on the image
            cv2.rectangle(img_copy, bbox[0], bbox[1], (0, 255, 0), 6)

    return result_rectangles, img_copy

def predict_on_test_images(test_files, data_path, window_scale, x_start_stop, y_start_stop, model, X_scaler, orientations, pix_per_cell, cell_per_block, spatial_size, hist_bins, threshold_value):
    """Predict car locations on a set of test images using a trained classifier and saves the predictions to a CSV.

    Args:
        test_files (list): List of test image file names.
        data_path (str): Path to the directory containing test images.
        window_scale (tuple): The scales of the windows to slide over the images.
        x_start_stop (tuple): The start and stop positions in the x-dimension to search for windows.
        y_start_stop (list): List of start and stop positions in the y-dimension for each scale.
        model (classifier): The trained classifier used to detect cars.
        X_scaler (scaler): The scaler used to normalize the feature vectors.
        orientations (int): Number of HOG orientations.
        pix_per_cell (int): Number of pixels per cell for the HOG features.
        cell_per_block (int): Number of cells per block for the HOG features.
        spatial_size (tuple): Spatial size for the spatial binning of the color channels.
        hist_bins (int): Number of histogram bins for color histogram features.
        threshold_value (float): Adaptive threshold value for heatmap processing.

    Returns:
        tuple: Contains lists of result images, bounding boxes, heatmap images, thresholded heatmap images, and result images with all bounding boxes.
    """
    result_images = []
    result_boxes = []
    heatmap_images = []
    threshold_images = []
    result_img_all_boxes = []
    rows = []

    for file_name in test_files:
        img_path = os.path.join(data_path, file_name)
        img = mpimg.imread(img_path)

        rectangles = get_rectangles(img, window_scale, x_start_stop, y_start_stop, model, X_scaler, orientations, pix_per_cell, cell_per_block, spatial_size, hist_bins)
        result_img_all_boxes.append(draw_boxes(img, rectangles, color=(0, 0, 255), thick=3))
        heatmap_image = np.zeros_like(img[:, :, 0])
        heatmap_image = add_heat(heatmap_image, rectangles)
        heatmap_images.append(heatmap_image)
        threshold_image = apply_adaptive_threshold(heatmap_image, threshold_value)
        threshold_images.append(threshold_image)
        labels = label(threshold_image)
        rectangles, result_image = draw_labeled_bboxes(img, labels)
        result_images.append(result_image)
        result_boxes.append(rectangles)

        # Convert bounding boxes to binary mask and produce run-length encoding
        rle = u.run_length_encoding(u.bounding_boxes_to_mask(rectangles, img.shape[0], img.shape[1]))

        # Append row to dataframe
        rows.append(['test/' + file_name, rle])

    # Create dataframe and save to csv file
    df_prediction = pd.DataFrame(columns=['Id', 'Predicted'], data=rows).set_index('Id')
    df_prediction.to_csv('sample_submission.csv')

    return result_images, result_boxes, heatmap_images, threshold_images, result_img_all_boxes
