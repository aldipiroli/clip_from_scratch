import os
import zipfile

import requests
from tqdm import tqdm


def download_file(url, dest):
    print(f"Downloading {url}...")
    response = requests.get(url, stream=True)
    total = int(response.headers.get("content-length", 0))
    with open(dest, "wb") as file, tqdm(
        desc=dest,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)
    print(f"Downloaded {dest}")


def extract_zip(zip_path, extract_to):
    print(f"Extracting {zip_path} to {extract_to} ...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted to {extract_to}")


def download_flickr8k_dataset(root_path):
    dest_path = os.path.join(root_path, "flickr8k")
    if os.path.exists(dest_path):
        print(f"Folder '{dest_path}' already exists. Skipping download.")
        return dest_path
    os.makedirs(dest_path, exist_ok=True)

    images_url = "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip"
    captions_url = "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip"

    images_zip = os.path.join(dest_path, "Flickr8k_Dataset.zip")
    captions_zip = os.path.join(dest_path, "Flickr8k_text.zip")

    download_file(images_url, images_zip)
    download_file(captions_url, captions_zip)

    extract_zip(images_zip, os.path.join(dest_path, "images"))
    extract_zip(captions_zip, os.path.join(dest_path, "captions"))

    os.remove(images_zip)
    os.remove(captions_zip)
    print("Cleanup done, zip files removed.")
    return dest_path
