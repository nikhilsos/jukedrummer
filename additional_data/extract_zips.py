# code to extract datas from a folder containing multiple zip files to a single folder

import os
import zipfile
from tqdm import tqdm

# Define the path to the folder containing the zip files
zip_folder_path = "/home/nikhil/jukedrummer/additional_data/zip_files"

# Define the path to the folder where you want to extract the files
extract_folder_path = "/home/nikhil/jukedrummer/additional_data/unzipped"

# Loop through all the files in the zip folder
for file_name in os.listdir(zip_folder_path):
    # Check if the file is a zip file
    if file_name.endswith(".zip"):
        # Define the path to the zip file
        zip_file_path = os.path.join(zip_folder_path, file_name)

        # Open the zip file
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            # Extract all the files in the zip file to the extract folder
            zip_ref.extractall(extract_folder_path)

print("All files have been extracted to the extract folder.")