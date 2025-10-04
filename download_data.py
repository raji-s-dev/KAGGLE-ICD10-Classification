import os
import gdown

# Google Drive folder URL
folder_url = "https://drive.google.com/drive/folders/1q_LOeGmRC0b33faVXxJIQUMmpODC9Ayh?usp=sharing"

# Destination folder (current working directory)
dest_folder = os.getcwd()

# Download entire folder
print("Downloading dataset from Google Drive...")
gdown.download_folder(url=folder_url, output=dest_folder, quiet=False, use_cookies=False)

print(f"All files downloaded to {dest_folder}")
