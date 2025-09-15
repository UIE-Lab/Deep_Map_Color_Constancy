# -*- coding: utf-8 -*-
"""
This script implements a color constancy algorithm based on the approach
by Ebner and Hansen, utilizing an RGB image and its corresponding depth map.
The goal is to correct the colors in an image by estimating and removing the
effect of the scene's illuminant.

Based on the paper: "Depth map color constancy" by Marc Ebner and Johannes Hansen.

Required Files in the same directory:
- 'im.png': The input RGB image to be corrected.
- 'im.npy': A numpy file containing the depth information for the image.
"""
import cv2
import numpy as np
import sys
import os
import matplotlib.pyplot as plt

def srgb_to_linear(srgb_image: np.ndarray) -> np.ndarray:
    """
    Converts an sRGB image (float, 0-1 range) to a linear RGB color space.
    This conversion is necessary for physically accurate light calculations.
    """
    mask = srgb_image <= 0.04045
    linear_image = np.zeros_like(srgb_image, dtype=np.float32)

    # Linear transformation for low-light values
    linear_image[mask] = srgb_image[mask] / 12.92
    # Gamma correction for high-light values
    linear_image[~mask] = np.power((srgb_image[~mask] + 0.055) / 1.055, 2.4)

    return linear_image


def linear_to_srgb(linear_image: np.ndarray) -> np.ndarray:
    """
    Converts a linear RGB image (float, 0-1 range) back to the sRGB color space
    for correct display on a monitor.
    """
    mask = linear_image <= 0.0031308
    srgb_image = np.zeros_like(linear_image, dtype=np.float32)

    # Linear transformation for low-light values
    srgb_image[mask] = linear_image[mask] * 12.92
    # Inverse gamma correction for high-light values
    srgb_image[~mask] = 1.055 * np.power(linear_image[~mask], 1.0 / 2.4) - 0.055

    return srgb_image


def apply_depth_aware_color_constancy(
    rgb_image: np.ndarray,
    depth_map: np.ndarray,
    sigma: float = 0.25,
    epsilon: float = 0.1,
    iterations: int = 300,
    kernel_size: int = 3,
    convergence_threshold: float = 1e-6
) -> tuple[np.ndarray, np.ndarray]:
    """
    Applies the depth-aware color constancy algorithm from Ebner and Hansen's paper.

    Args:
        rgb_image: The input sRGB image as a NumPy array (H, W, 3).
        depth_map: The corresponding depth map, normalized to the [0, 1] range.
        sigma: Parameter controlling the spatial extent of the averaging.
        epsilon: Depth threshold to prevent averaging across discontinuities.
        iterations: The maximum number of iterations for the illuminant estimation.
        kernel_size: The size of the neighborhood window for averaging (e.g., 3 for 3x3).
        convergence_threshold: Stops iteration if the change is smaller than this value.

    Returns:
        A tuple containing:
        - The final corrected RGB image (uint8).
        - The estimated illuminant map for visualization (uint8).
    """
    # --- STEP 1: PRE-PROCESSING AND COLOR SPACE CONVERSION ---
    # Normalize image from [0, 255] uint8 to [0, 1] float32
    srgb_float_image = rgb_image.astype(np.float32) / 255.0
    # Convert from sRGB to Linear RGB for physically correct calculations
    linear_image = srgb_to_linear(srgb_float_image)
    # Switch to the logarithmic domain to simplify calculations (I = R*L -> log(I) = log(R)+log(L))
    log_image = np.log(linear_image + 1e-9)  # Add epsilon to avoid log(0)

    # --- STEP 2: ILLUMINANT ESTIMATION VIA LOCAL SPACE AVERAGE (LSA) ---
    height, width = log_image.shape[:2]
    max_dimension = max(width, height)
    # The 'p' parameter determines the blending between the original image and the neighborhood average
    p = 1.0 / ((sigma * max_dimension)**2 + 1)
    
    # Initialize the illuminant estimate with the log image itself
    log_illuminant_estimate = np.copy(log_image)
    neighborhood_avg = np.zeros_like(log_illuminant_estimate)
    pad_width = kernel_size // 2

    print(f"Parameter 'p' value: {p:.6f}")
    print(f"Starting iterations (max: {iterations}, convergence threshold: {convergence_threshold})...")

    # Iteratively estimate the illuminant
    for i in range(iterations):
        prev_log_illuminant_estimate = log_illuminant_estimate.copy()
        
        # Pad arrays to handle borders
        padded_illuminant = cv2.copyMakeBorder(log_illuminant_estimate, pad_width, pad_width, pad_width, pad_width, cv2.BORDER_REPLICATE)
        padded_depth = cv2.copyMakeBorder(depth_map, pad_width, pad_width, pad_width, pad_width, cv2.BORDER_REPLICATE)

        # Calculate the neighborhood average for each pixel
        for x in range(height):
            for y in range(width):
                center_depth = padded_depth[x + pad_width, y + pad_width]
                
                # Get the local neighborhood windows
                neighborhood_illuminant = padded_illuminant[x : x + kernel_size, y : y + kernel_size]
                neighborhood_depth = padded_depth[x : x + kernel_size, y : y + kernel_size]
                
                # --- Create masks to select valid neighbors ---
                # 1. Mask to exclude the center pixel from its own average
                center_mask = np.ones(neighborhood_illuminant.shape[:2], dtype=bool)
                center_mask[pad_width, pad_width] = False
                
                # 2. Mask for neighbors with a depth difference smaller than the threshold
                depth_diff_mask = np.abs(neighborhood_depth - center_depth) <= epsilon
                
                # 3. Combine masks: must be a neighbor (not center) AND have a similar depth
                final_mask = depth_diff_mask & center_mask

                valid_neighbors = neighborhood_illuminant[final_mask]
                
                # Calculate the mean of the valid neighbors
                if len(valid_neighbors) > 0:
                    neighborhood_avg[x, y] = np.mean(valid_neighbors, axis=0)
                else:
                    # If no valid neighbors, use the estimate from the previous iteration
                    neighborhood_avg[x, y] = log_illuminant_estimate[x, y]

        # Update the illuminant estimate using the blending formula from the paper
        log_illuminant_estimate = log_image * p + neighborhood_avg * (1 - p)

        # Check for convergence
        diff = np.mean(np.abs(log_illuminant_estimate - prev_log_illuminant_estimate))
        print(f"  Iteration {i+1}/{iterations} -> Mean Change: {diff:.8f}")
        if diff < convergence_threshold:
            print(f"Convergence reached at iteration {i+1}. Stopping.")
            break
            
    print("Iterations finished.")

    # --- STEP 3: FINAL IMAGE CREATION AND NORMALIZATION ---
    # Calculate the reflectance map: R = I / (2*L) as specified in the paper's text for Figure 3
    # In log space: log(R) = log(I) - log(2*L) = log(I) - (log(L) + log(2))
    reflectance_log = log_image - (log_illuminant_estimate + np.log(2))
    
    # Convert reflectance back from log to linear space
    reflectance_linear = np.exp(reflectance_log)
    
    # Convert back to sRGB for display
    result_srgb = linear_to_srgb(reflectance_linear)
    
    # Clip and convert to uint8 for saving/displaying
    corrected_image = np.clip(result_srgb, 0, 1)
    corrected_image = (corrected_image * 255).astype(np.uint8)

    # Prepare the illuminant map for visualization
    illuminant_linear = np.exp(log_illuminant_estimate)
    illuminant_vis_srgb = linear_to_srgb(illuminant_linear)
    illuminant_vis = np.clip(illuminant_vis_srgb, 0, 1)
    illuminant_map = (illuminant_vis * 255).astype(np.uint8)

    return corrected_image, illuminant_map

if __name__ == '__main__':
    # File paths
    rgb_file = 'im.png'
    depth_file = 'im.npy'

    # Check if required files exist
    if not os.path.exists(rgb_file) or not os.path.exists(depth_file):
        print(f"Error: Make sure '{rgb_file}' and '{depth_file}' exist in the current directory.")
        sys.exit()

    # Read the image and convert from OpenCV's BGR format to RGB
    bgr_image = cv2.imread(rgb_file)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    # Load the depth map and normalize it to the [0, 1] range
    depth_data = np.load(depth_file)
    min_depth, max_depth = depth_data.min(), depth_data.max()
    normalized_depth_map = (depth_data - min_depth) / (max_depth - min_depth)

    # Resize images to speed up processing for this demonstration
    new_size = (200, 160)
    rgb_image_resized = cv2.resize(rgb_image, new_size, interpolation=cv2.INTER_AREA)
    depth_map_resized = cv2.resize(normalized_depth_map, new_size, interpolation=cv2.INTER_LINEAR)

    # Call the main color constancy function
    corrected_image, illuminant_map = apply_depth_aware_color_constancy(
        rgb_image_resized,
        depth_map_resized
    )

    # --- VISUALIZE THE RESULTS ---
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle('Depth-Aware Color Constancy Results', fontsize=16)

    axes[0].imshow(rgb_image_resized)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(depth_map_resized, cmap='gray')
    axes[1].set_title('Normalized Depth Map')
    axes[1].axis('off')

    axes[2].imshow(illuminant_map)
    axes[2].set_title('Estimated Illuminant')
    axes[2].axis('off')

    axes[3].imshow(corrected_image)
    axes[3].set_title('Corrected Image')
    axes[3].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()