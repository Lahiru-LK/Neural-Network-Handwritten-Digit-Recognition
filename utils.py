import numpy as np
from scipy.ndimage import center_of_mass, gaussian_filter
from skimage.transform import resize

def preprocess_digit(image):
    # Center the digit
    y_center, x_center = center_of_mass(image)
    shift_y = int(14 - y_center)
    shift_x = int(14 - x_center)
    centered_image = np.roll(image, shift_y, axis=0)
    centered_image = np.roll(centered_image, shift_x, axis=1)

    # Resize to 28x28 to match MNIST
    resized_image = resize(centered_image, (28, 28), anti_aliasing=True, mode='constant')

    # Normalize pixel values
    normalized_image = resized_image / np.max(resized_image) if np.max(resized_image) > 0 else resized_image

    # Apply Gaussian smoothing
    smoothed_image = gaussian_filter(normalized_image, sigma=0.8)

    return smoothed_image.flatten()
