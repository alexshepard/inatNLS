import hashlib
import os

from pathlib import Path
from PIL import Image
import requests


class ImageManager:
    def __init__(self, cache_dir):
        self.cache_dir = Path(cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)

    def download_image(self, photo_url, local_path):
        if not os.path.exists(local_path):
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            response = requests.get(photo_url)
            if response.status_code == 200:
                with open(local_path, "wb") as file:
                    file.write(response.content)
            return response.status_code == 200
        return False

    def open_image(self, path):
        return Image.open(path)

    def path_for_photo_id(self, photo_id):
        msg = str(photo_id).encode("utf-8")
        m = hashlib.sha256()
        m.update(msg)
        hashed = m.hexdigest()
        filename = "{}.jpg".format(photo_id)
        photo_paths = [self.cache_dir, hashed[0:2], hashed[2:4], filename]
        return os.path.join(*photo_paths)

    def url_for_photo_id(self, photo_id, extension):
        image_base_url = (
            "https://inaturalist-open-data.s3.amazonaws.com/photos/{}/medium.{}"
        )
        return image_base_url.format(
            photo_id, extension
        )
