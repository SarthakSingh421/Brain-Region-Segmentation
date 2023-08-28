import os
import nibabel as nib
import numpy as np
import cv2

# Directory paths
cta_directory = r"C:/Proxmed/Extracted/CTA"
mask_directory = r"C:/Proxmed/Extracted/Brain Masks"
output_cta_directory = r"C:/Proxmed/CTA_MIPs"
output_mask_directory = r"C:/Proxmed/Mask_MIPs"

# Create output directories if they don't exist
os.makedirs(output_cta_directory, exist_ok=True)
os.makedirs(output_mask_directory, exist_ok=True)

# Iterate through CTA images
for cta_filename in os.listdir(cta_directory):
    cta_path = os.path.join(cta_directory, cta_filename)
    if cta_filename.endswith('.nii.gz'):
        cta_basename = os.path.splitext(cta_filename)[0]
        
        try:
            # Load CTA image
            cta_image = nib.load(cta_path)
            cta_data = cta_image.get_fdata()
            
            # Calculate Maximum Intensity Projection (MIP)
            cta_mip = np.max(cta_data, axis=2)
            
            # Normalize and convert to CV_8U
            cta_mip_normalized = cv2.normalize(cta_mip, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            
            # Save the MIP image
            output_cta_filename = f"{cta_basename}_MIP.png"
            output_cta_path = os.path.join(output_cta_directory, output_cta_filename)
            cv2.imwrite(output_cta_path, cta_mip_normalized)
            
        except Exception as e:
            print(f"Error processing {cta_filename}: {e}")
            continue

# Iterate through brain masks
for mask_filename in os.listdir(mask_directory):
    mask_path = os.path.join(mask_directory, mask_filename)
    if mask_filename.endswith('.nii.gz'):
        mask_basename = os.path.splitext(mask_filename)[0]
        
        try:
            # Load brain mask
            mask_image = nib.load(mask_path)
            mask_data = mask_image.get_fdata()
            
            # Calculate Maximum Intensity Projection (MIP)
            mask_mip = np.max(mask_data, axis=2)
            
            # Normalize and convert to CV_8U
            mask_mip_normalized = cv2.normalize(mask_mip, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            
            # Save the MIP image
            output_mask_filename = f"{mask_basename}_MIP.png"
            output_mask_path = os.path.join(output_mask_directory, output_mask_filename)
            cv2.imwrite(output_mask_path, mask_mip_normalized)
            
        except Exception as e:
            print(f"Error processing {mask_filename}: {e}")
            continue

print("MIP images (normalized) generated and saved successfully.")
