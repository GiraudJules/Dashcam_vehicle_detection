import glob

import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


def data_information(cars, notcars, train_cars, test_cars, train_notcars, test_notcars, img):
    """Print out statistics about the dataset.

    Args:
        cars (list): List of car images.
        notcars (list): List of non-car images.
        train_cars (list): List of car images used for training.
        test_cars (list): List of car images used for testing.
        train_notcars (list): List of non-car images used for training.
        test_notcars (list): List of non-car images used for testing.
        img (np.array): Sample image to get shape and data type.

    Returns:
        None. The function prints out the dataset statistics.
    """
    print('We have in our list of images a count of:\n',
        len(cars), ' cars\n',
        len(notcars), ' non-cars')
    print('We have in our list of images a count of:\n',
        len(train_cars), ' train cars\n',
        len(test_cars), ' test cars')
    print('We have in our list of images a count of:\n',
        len(train_notcars), ' train non-cars\n',
        len(test_notcars), ' test non-cars')
    print('The size of the images is:\n',
        img.shape)
    print('The data type is:\n',
        img.dtype)


def visualize_cars_and_ncars_images(images, num_cols, title):
    """Visualizes a subset of car and non-car images in a grid.

    Args:
        images (list): List of images to visualize.
        num_cols (int): Number of columns for the display grid.
        title (str): Title for the visualization.

    Returns:
        None. The function displays the images using matplotlib.
    """
    fig, axes = plt.subplots((len(images) + 1) // num_cols, num_cols, figsize=(20, 5))
    fig = plt.gcf()
    fig.canvas.manager.set_window_title(title)
    print(title)
    for ax, image in zip(axes.flat, images):
        ax.imshow(image)
    plt.show()


def examples_hog_plot(car_img, car_hog, notcar_img, notcar_hog):
    """Visualize car and non-car images alongside their HOG visualizations.

    Args:
        car_img (np.array): The car image to be visualized.
        car_hog (np.array): HOG visualization of the car image.
        notcar_img (np.array): The non-car image to be visualized.
        notcar_hog (np.array): HOG visualization of the non-car image.

    Returns:
        None. The function visualizes the images using matplotlib.
    """
    _, ax = plt.subplots(1, 4, figsize=(20, 5))
    ax[0].imshow(car_img)
    ax[0].set_title('Example of car Image')
    ax[1].imshow(car_hog, cmap='gray')
    ax[1].set_title('car HOG Visualization')
    ax[2].imshow(notcar_img, cmap='gray')
    ax[2].set_title('Example of not-car Image')
    ax[3].imshow(notcar_hog, cmap='gray')
    ax[3].set_title('not car HOG Visualization')
    plt.show()


def show_test_image(name_dir, index):
    """
    Display a single test image from a directory.

    Args:
        name_dir (str): The file path for the directory containing the test images, with /*.jpg at the end.
        index (int): The index of the test image to display.

    Returns:
        test_images: The test images as a numpy array.
    """
    test_files = glob.glob(name_dir)
    test_images = []
    for file in test_files:
        test_image = cv2.imread(file)
        test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
        test_images.append(test_image)
    test_images = np.asarray(test_images)
    print("Test images shape is:", test_images.shape)
    print("Test image shape is:", test_images[index].shape)
    plt.imshow(test_images[index])
    plt.title(f"Test Image {index+1}")
    plt.show()
    return test_images, test_images[index]


def load_test_images(dir):
    """Load test images from a directory.

    Args:
        dir (str): The directory path containing the test images.

    Returns:
        list: A list of test images loaded from the directory.
    """
    test_images = []
    images = glob.glob(dir + '*.jpg')
    for image in images:
        test_images.append(mpimg.imread(image))
    return test_images


def detector_results(result_images, heatmap_images, threshold_images, result_img_all_boxes, index):
    """Visualize the results from the vehicle detection process.

    Args:
        result_images (list): List of final detection result images.
        heatmap_images (list): List of heatmap images.
        threshold_images (list): List of thresholded heatmap images.
        result_img_all_boxes (list): List of images with all detected boxes.
        index (int): Index of the image to be visualized.

    Returns:
        None. The function visualizes the images using matplotlib.
    """
    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 10))

    # Plot first image in row 0, column 0
    axes[0].imshow(result_img_all_boxes[index])
    axes[0].set_title('All boxes Image')

    # Plot second image in row 0, column 1
    axes[1].imshow(heatmap_images[index], cmap='hot', vmin=0, vmax=np.amax(heatmap_images[index]))
    axes[1].set_title('Heatmap image')

    # Plot third image in row 1, column 0
    axes[2].imshow(threshold_images[index], cmap='hot', vmin=0, vmax=np.amax(heatmap_images[index]))
    axes[2].set_title('Thresholded Heatmap Image')

    # Plot fourth image in row 1, column 1
    axes[3].imshow(result_images[index])
    axes[3].set_title('Result Image')

    plt.tight_layout()
    plt.show()


def visualize_images(input_images, num_cols, figure_name, cmap=None):
    """Display a list of images in a grid layout.

    Args:
        input_images (list): List of images to be visualized.
        num_cols (int): Number of columns for the grid layout.
        figure_name (str): Title for the overall visualization.
        cmap (str, optional): Color map to be used for visualization. Defaults to None.

    Returns:
        None. The function visualizes the images using matplotlib.
    """
    fig, axes = plt.subplots((int)((len(input_images) + 1) / num_cols), num_cols, figsize=(24, 20))

    fig = plt.gcf()
    fig.suptitle(figure_name)

    print(figure_name)

    for ax, image in zip(axes.flat, input_images):
        if (cmap == "gray" or cmap == 'hot'):
            ax.imshow(image, cmap=cmap)
        elif (image.shape[2] == 1):
            ax.imshow(image[:, :, 0], cmap=cmap)
        else:
            ax.imshow(image, cmap=cmap)

    plt.show()
