import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

# Kaggle API 인증 - 개인 키 입력
os.environ['KAGGLE_USERNAME'] = "your_username"
os.environ['KAGGLE_KEY'] = "your_key"

# 다운로드 경로 설정
dataset_zip = "chest-xray-pneumonia.zip"
extract_path = "chest_xray"

# Kaggle API 설정
api = KaggleApi()
api.authenticate()

# 데이터셋 다운로드
if not os.path.exists(dataset_zip):
    print("Downloading dataset...")
    api.dataset_download_files("paultimothymooney/chest-xray-pneumonia", path=".", unzip=False)
else:
    print("Dataset already exists. Skipping download.")

# 압축 해제
if not os.path.exists(extract_path):
    print("Extracting dataset...")
    with zipfile.ZipFile(dataset_zip, "r") as zip_ref:
        zip_ref.extractall(extract_path)
    print("Extraction complete.")
else:
    print("Dataset already extracted. Skipping extraction.")
