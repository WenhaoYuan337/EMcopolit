import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import ellipse
from skimage.transform import rotate
import random


# Function to check if a new ellipse overlaps with existing particles
def is_overlapping(new_rr, new_cc, current_image):
    overlap = np.sum(current_image[new_rr, new_cc]) > 0
    return overlap


# Function to generate ellipses with accurate size and ellipticity
def generate_random_rotated_ellipse_with_min_diff_ellipticity(avg_diameter, size_variability, max_ellipticity,
                                                              ellipticity_variability, image_size):
    # Generate size variability (0 = same size, 1 = highly varied sizes)
    size_variation_factor = np.random.uniform(1 - size_variability, 1 + size_variability)  # Variability factor
    r_radius = (avg_diameter / 2) * size_variation_factor  # Base radius with variability applied

    # Compute ellipticity based on ellipticity_variability (0 = circle, 1 = max_ellipticity)
    ellipticity_ratio = np.random.uniform(0, ellipticity_variability)
    current_ellipticity = 1 - ellipticity_ratio * (1 - max_ellipticity)  # Scale ellipticity within the desired range
    c_radius = r_radius * current_ellipticity  # Adjust the minor axis based on ellipticity

    # Ensure radii are not too small or zero, set a minimum value (e.g., 3 pixels)
    min_radius = 3
    r_radius = max(r_radius, min_radius)
    c_radius = max(c_radius, min_radius)

    # Ensure there is a minimum difference between radii to avoid square-like shapes
    min_difference_ratio = 0.3  # Set a minimum ratio difference between r_radius and c_radius (e.g., 30%)
    if abs(r_radius - c_radius) < (r_radius * min_difference_ratio):
        if r_radius > c_radius:
            c_radius = r_radius * (1 - min_difference_ratio)
        else:
            r_radius = c_radius * (1 - min_difference_ratio)

    # Set a random center within the bounds of the image
    center_y = random.randint(int(r_radius), image_size[0] - int(r_radius))
    center_x = random.randint(int(c_radius), image_size[1] - int(c_radius))

    # Generate ellipse coordinates
    rr, cc = ellipse(center_y, center_x, int(r_radius), int(c_radius), image_size)

    # Create a mask for the ellipse
    mask = np.zeros(image_size, dtype=np.uint8)
    mask[rr, cc] = 255

    # Apply a random rotation to the mask
    angle = random.uniform(0, 360)  # Random angle between 0 and 360 degrees
    rotated_mask = rotate(mask, angle, resize=False, mode='constant', cval=0, preserve_range=True).astype(np.uint8)

    return rotated_mask


# Function to generate an image with non-overlapping, randomly rotated ellipses with optimized size and ellipticity
def generate_nanocluster_image_with_min_diff_ellipticity(avg_diameter, num_particles, size_variability, max_ellipticity,
                                                         ellipticity_variability, image_size, max_attempts=100):
    image = np.zeros(image_size, dtype=np.uint8)

    for _ in range(num_particles):
        attempts = 0
        while attempts < max_attempts:
            rotated_mask = generate_random_rotated_ellipse_with_min_diff_ellipticity(avg_diameter, size_variability,
                                                                                     max_ellipticity,
                                                                                     ellipticity_variability,
                                                                                     image_size)
            # Check for overlap with existing particles
            if not is_overlapping(rotated_mask.nonzero()[0], rotated_mask.nonzero()[1], image):
                image += rotated_mask
                break
            attempts += 1

    return image


# Ensure the generated_mask folder exists
output_folder = "generated_mask"
os.makedirs(output_folder, exist_ok=True)

# Parameters definition
scaling_factor = 512 / 2048 # Scaling factor for diameter based on image size change from 2048 to 512
avg_diameter = 60 * scaling_factor  # Average particle diameter in pixels
num_particles = 100  # Number of particles to generate
size_variability = 0.8  # Control the spread of particle sizes (1 = same size, 0 = highly varied)
max_ellipticity = 0.7  # Maximum ellipticity (0.7 means max difference between axes is 30%)
ellipticity_variability = 0.1  # Control how many particles are more elliptical (0 = all circles, 1 = max_ellipticity range)
image_size = (512, 512)  # Size of the image
num_images = 10  # Number of images to generate

# Generate and save multiple images
for i in range(num_images):
    image = generate_nanocluster_image_with_min_diff_ellipticity(avg_diameter, num_particles, size_variability,
                                                                 max_ellipticity, ellipticity_variability, image_size)

    # Save the image to the generated_mask folder
    plt.imsave(f"{output_folder}/mask_{i + 1}.png", image, cmap='gray')
