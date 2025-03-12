import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

# Kaggle API authentication - Enter your personal key
os.environ['KAGGLE_USERNAME'] = "your_username"
os.environ['KAGGLE_KEY'] = "your_key"

# Set download paths
dataset_zip = "chest-xray-pneumonia.zip"
extract_path = "chest_xray"

# Initialize Kaggle API
api = KaggleApi()
api.authenticate()

# Download dataset
if not os.path.exists(dataset_zip):
    print("Downloading dataset...")
    api.dataset_download_files("paultimothymooney/chest-xray-pneumonia", path=".", unzip=False)
else:
    print("Dataset already exists. Skipping download.")

# Extract dataset
if not os.path.exists(extract_path):
    print("Extracting dataset...")
    with zipfile.ZipFile(dataset_zip, "r") as zip_ref:
        zip_ref.extractall(extract_path)
    print("Extraction complete.")
else:
    print("Dataset already extracted. Skipping extraction.")
