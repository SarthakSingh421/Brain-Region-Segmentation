import os
import shutil

# Path to the main folder 
main_folder = r'C:\Proxmed\Extracted'

# Paths to the new folders for CTA images and brain masks
cta_folder = r'C:\Proxmed\Extracted\CTA'
mask_folder = r'C:\Proxmed\Extracted\Brain masks'

# Iterate through subfolders
for subfolder_name in os.listdir(main_folder):
    subfolder_path = os.path.join(main_folder, subfolder_name)
    if os.path.isdir(subfolder_path):
        cta_file = None
        mask_file = None
        
        # Find CTA image and mask files 
        for file_name in os.listdir(subfolder_path):
            if file_name.endswith('.nii.gz'):
                if 'ROI' in file_name:
                    mask_file = os.path.join(subfolder_path, file_name)
                else:
                    cta_file = os.path.join(subfolder_path, file_name)
        
        # Move files to the respective folders
        if cta_file:
            cta_dest = os.path.join(cta_folder, os.path.basename(cta_file))
            if os.path.exists(cta_dest):
                cta_dest = cta_dest.replace('.nii.gz', '_duplicate.nii.gz')
            shutil.copy(cta_file, cta_dest)
        if mask_file:
            mask_dest = os.path.join(mask_folder, os.path.basename(mask_file))
            if os.path.exists(mask_dest):
                mask_dest = mask_dest.replace('.nii.gz', '_duplicate.nii.gz')
            shutil.copy(mask_file, mask_dest)

print("Files moved successfully.")
