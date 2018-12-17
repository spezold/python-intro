"""
In this lesson, we will
- create a 2D plot of a 1D sine wave
- create a 3D plot of a 2D sine wave
- load an image and apply a median filter
"""

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import skimage

def draw_sine_2d(amplitude=1, num_values=1000, x_min=0, x_max=2 * np.pi):
    """
    Draw a sine wave.
    
    :param amplitude: The amplitude of the wave
    :param num_values: How many values to sample on the given interval
    :param x_min: Lower boundary of the interval to be drawn (inclusive)
    :param x_max: Upper boundary of the interval to be drawn (inclusive)
    """
    
    x_s = np.linspace(x_min, x_max, num_values)
    y_s = amplitude * np.sin(x_s)
    plt.plot(x_s, y_s)
    plt.show()
    

def draw_sine_3d(amplitude=1, num_x_values=1000, num_y_values=1000,
                 x_min=-2 * np.pi, x_max=2 * np.pi,
                 y_min=-2 * np.pi, y_max=2 * np.pi):
    """
    Draw a sine wave on the 2D plane, spreading out radially from the origin.
    
    :param amplitude: The amplitude of the wave
    :param num_x_values: How many values to sample on the given x axis interval
    :param num_y_values: How many values to sample on the given y_axis interval
    :param x_min: Lower x axis boundary of the interval to be drawn (inclusive)
    :param x_max: Upper x axis boundary of the interval to be drawn (inclusive)
    :param y_min: Lower y axis boundary of the interval to be drawn (inclusive)
    :param y_max: Upper y axis boundary of the interval to be drawn (inclusive)
    """
    
    x_s = np.linspace(x_min, x_max, num_x_values)[:, np.newaxis]
    y_s = np.linspace(y_min, y_max, num_y_values)[np.newaxis, :]
    r_s = np.sqrt(x_s ** 2 + y_s ** 2)
    z_s = amplitude * np.sin(r_s)
    
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    surf = ax.plot_surface(x_s, y_s, z_s, cmap=cm.coolwarm)
    fig.colorbar(surf)
    plt.show()
    

def load_sample_image(degrade=True):
    """
    Load the sample image "python.png".
    
    :param degrade: If True, add a bit of salt and pepeer noise.
    :return: The sample image as a 512x512 Numpy array, with values in [0, 1].
    """
    img = skimage.io.imread("python.png") / 255.
    if degrade:
        img = skimage.util.random_noise(img, mode="s&p", seed=1337)
    return img
    


def filter_image1(img, radius=1):
    """
    Filter the given image with a median filter of given radius, using our own,
    slow implementation.
    
    :param img: 2D Numpy array
    :param radius: Filter size will be a square of side length `2 * radius + 1`
    """
    
    result = img.copy()
    
    for i in range(radius, img.shape[0] - radius):
        for j in range(radius, img.shape[1] - radius):
            
            result[i, j] = np.median(img[i - radius : i + radius + 1,
                                         j - radius : j + radius + 1])
    
    return result


def filter_image2(img, radius=1):
    """
    Filter the given image with a median filter of given radius, using the
    implementation from `skimage.filters`.
    
    :param img: 2D Numpy array
    :param radius: Filter size will be a square of side length `2 * radius + 1`
    """
    flt = np.ones((2 * radius + 1, 2 * radius + 1))
    result = skimage.filters.median(img, flt) / 255.

    return result

    
if __name__ == "__main__":

    draw_sine_2d(3)
    draw_sine_3d(5)
    
    img_degraded = load_sample_image()
    img_original = load_sample_image(degrade=False)
    
    img_filtered1 = filter_image1(img_degraded)
    img_filtered2 = filter_image2(img_degraded)
    
    plt.figure()
    plt.subplot(141)
    plt.imshow(img_original, cmap="gray")
    plt.title("Original (noise-free) image")
    plt.subplot(142)
    plt.imshow(img_degraded, cmap="gray")
    plt.title("Image with salt-and-pepper noise")
    plt.subplot(143)
    plt.imshow(img_filtered1, cmap="gray")
    plt.title("Restored image (own implementation)")
    plt.subplot(144)
    plt.imshow(img_filtered2, cmap="gray")
    plt.title("Restored image (skimage implementation)")