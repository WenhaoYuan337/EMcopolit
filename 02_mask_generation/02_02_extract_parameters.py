import os
import numpy as np
from skimage import io, measure
from scipy.stats import variation

# Folder containing the mask images
mask_folder = 'F:/Seg/EMcopolit/label'


# Function to process each image and extract parameters
def process_mask_image(image_path):
    mask_image = io.imread(image_path)
    labeled_mask = measure.label(mask_image)
    properties = measure.regionprops(labeled_mask)

    diameters = []
    ellipticities = []

    for prop in properties:
        diameters.append(prop.equivalent_diameter)

        if prop.major_axis_length > 0:
            ellipticity = 1 - (prop.minor_axis_length / prop.major_axis_length)
            ellipticities.append(ellipticity)

    avg_diameter = np.mean(diameters)
    num_particles = len(diameters)
    size_variability = variation(diameters)
    max_ellipticity = np.max(ellipticities)
    ellipticity_variability = variation(ellipticities)

    return {
        'avg_diameter': avg_diameter,
        'num_particles': num_particles,
        'size_variability': size_variability,
        'max_ellipticity': max_ellipticity,
        'ellipticity_variability': ellipticity_variability
    }


# List all mask images in the folder
mask_images = [f for f in os.listdir(mask_folder) if f.endswith('.png') or f.endswith('.jpg')]

# Process each image and collect statistics
results = {}
for mask_image in mask_images:
    image_path = os.path.join(mask_folder, mask_image)
    results[mask_image] = process_mask_image(image_path)

# Output the results
print(results)
