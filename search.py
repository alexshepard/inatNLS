import csv
import hashlib
import logging
import os
from pathlib import Path
import yaml

from elasticsearch import Elasticsearch
import numpy as np
from PIL import Image
import requests
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

CONFIG = yaml.safe_load(open("config.yml"))


class Search:
    def __init__(self):
        self.model = SentenceTransformer(CONFIG["model_name"])
        self.es = Elasticsearch("http://localhost:9200")
        self.image_cache_path = Path(CONFIG["image_cache_dir"])
        os.makedirs(self.image_cache_path, exist_ok=True)
        client_info = self.es.info()
        print("Connected to Elasticsearch!")

    def get_embedding(self, text):
        # this is ugly but otherwise pytorch is logging constantly
        # haven't been able to track down which logger to disable
        for key in logging.Logger.manager.loggerDict:
            logging.getLogger(key).disabled = True

        return self.model.encode(text)

    def delete_index(self, index_name):
        self.es.indices.delete(index=index_name, ignore_unavailable=True)

    def create_index(self, index_name):
        self.es.indices.create(
            index=index_name,
            mappings={
                "properties": {
                    "embedding": {
                        "type": "dense_vector",
                    }
                }
            },
        )

    def insert_documents(self, documents, index_name, pbar):
        batch_insert_times = []

        operations = []
        for document in documents:
            local_path = self.path_for_photo_id(
                CONFIG["image_cache_dir"], document["photo_id"]
            )

            # if we don't have the image, try to download it
            image_base_url = (
                "https://inaturalist-open-data.s3.amazonaws.com/photos/{}/medium.{}"
            )
            photo_url = image_base_url.format(
                document["photo_id"], document["extension"]
            )

            if not os.path.exists(local_path):
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                r = requests.get(photo_url)
                if r.status_code == 200:
                    with open(local_path, "wb") as fd:
                        for chunk in r.iter_content(chunk_size=128):
                            fd.write(chunk)

            # if we still don't have the image, skip it
            if not os.path.exists(local_path):
                print("skipping {}".format(local_path))
                continue

            try:
                img = Image.open(local_path)
                img_emb = self.get_embedding(img)
                operations.append({"index": {"_index": index_name}})
                operations.append({**document, "embedding": img_emb})
            except:
                print("couldn't open or encode {}".format(local_path))
                continue

            pbar.update(1)

        return self.es.bulk(operations=operations)

    def add_to_index(self, index_name, data_file):
        batch_insert_times = []

        with open(data_file, "r") as file:
            num_docs = len(file.readlines()) - 1

        pbar = tqdm(
            total=num_docs,
            bar_format="{l_bar}{bar:30}{r_bar}{bar:-30b}",
            dynamic_ncols=True,
        )
        with open(data_file) as csvfile:
            csvreader = csv.DictReader(csvfile)
            documents = []
            for row in csvreader:
                documents.append(row)

                # insert in batches
                if len(documents) == CONFIG["insert_batch_size"]:
                    response = self.insert_documents(documents, index_name, pbar)
                    batch_insert_times.append(response["took"])
                    documents = []

        # insert anything at the end
        response = self.insert_documents(documents, index_name, pbar)
        batch_insert_times.append(response["took"])

        pbar.close()

        print(
            "indexed {} documents in {} batches with an average of {} ms per batch".format(
                num_docs, len(batch_insert_times), np.mean(batch_insert_times)
            )
        )

    def search(self, index_name, **query_args):
        return self.es.search(index=index_name, **query_args)

    def path_for_photo_id(self, base_dir, photo_id):
        msg = str(photo_id).encode("utf-8")
        m = hashlib.sha256()
        m.update(msg)
        hashed = m.hexdigest()
        filename = "{}.jpg".format(photo_id)
        photo_paths = [base_dir, hashed[0:2], hashed[2:4], filename]
        return os.path.join(*photo_paths)
