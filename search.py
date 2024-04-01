import json
from pprint import pprint
import os
import time
import csv
import logging
from pathlib import Path
import hashlib

from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from PIL import Image
import requests
from tqdm.auto import tqdm

load_dotenv()

# move to yaml
CONFIG = {
    "CLIP_MODEL": "clip-ViT-B-32",
    "BASE_IMAGE_URL": "https://inaturalist-open-data.s3.amazonaws.com/photos/{}/medium.{}",
    "DATA_FILE": "complete_100k_obs_sample.csv",
    "IMAGE_CACHE": "images/",
    "INSERT_BATCH_SIZE": 200,
}


class Search:
    def __init__(self):
        self.model = SentenceTransformer(CONFIG["CLIP_MODEL"])
        self.es = Elasticsearch("http://localhost:9200")
        self.image_cache_path = Path(CONFIG["IMAGE_CACHE"])
        os.makedirs(self.image_cache_path, exist_ok=True)
        client_info = self.es.info()
        print("Connected to Elasticsearch!")

    def get_embedding(self, text):
        for key in logging.Logger.manager.loggerDict:
            logging.getLogger(key).disabled = True

        return self.model.encode(text)

    def create_index(self, index_name):
        self.es.indices.delete(index=index_name, ignore_unavailable=True)
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

    def insert_document(self, document, index_name):
        return self.es.index(
            index=index_name,
            document={**document, "embedding": self.get_embedding(document["summary"])},
        )

    def insert_documents(self, documents, index_name):
        operations = []
        for document in tqdm(documents):
            local_path = self.path_for_photo_id(
                CONFIG["IMAGE_CACHE"], document["photo_id"]
            )

            # if we don't have the image, skip it
            if not os.path.exists(local_path):
                print("skipping {}".format(local_path))
                continue

            # need to embed in try catch and skip when we can't get an image
            img = Image.open(local_path)
            img_emb = self.get_embedding(img)
            operations.append({"index": {"_index": index_name}})
            operations.append({**document, "embedding": img_emb})
        return self.es.bulk(operations=operations)

    def reindex(self, index_name):
        self.create_index(index_name)
        with open(CONFIG["DATA_FILE"]) as csvfile:
            csvreader = csv.DictReader(csvfile)
            documents = []
            for row in csvreader:
                documents.append(row)

                # insert in batches
                if len(documents) == CONFIG["INSERT_BATCH_SIZE"]:
                    response = self.insert_documents(documents, index_name)
                    print(
                        "Index with {} documents created in {} milliseconds".format(
                            len(response["items"]), response["took"]
                        )
                    )
                    documents = []

        # insert anything at the end
        response = self.insert_documents(documents, index_name)
        print(
            "Index with {} documents created in {} milliseconds".format(
                len(response["items"]), response["took"]
            )
        )

    def search(self, index_name, **query_args):
        return self.es.search(index=index_name, **query_args)

    def retrieve_document(self, index_name, id):
        return self.es.get(index=index_name, id=id)

    def path_for_photo_id(self, base_dir, photo_id):
        msg = str(photo_id).encode("utf-8")
        m = hashlib.sha256()
        m.update(msg)
        hashed = m.hexdigest()
        filename = "{}.jpg".format(photo_id)
        photo_paths = [base_dir, hashed[0:2], hashed[2:4], filename]
        return os.path.join(*photo_paths)
