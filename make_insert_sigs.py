import os
import logging

from tqdm.auto import tqdm
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.http.models import PointStruct
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import numpy as np

for key in logging.Logger.manager.loggerDict:
    logging.getLogger(key).disabled = True

import logging.config
logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': True,
})


def main():
    client = QdrantClient("localhost", port=6333)
    #client.create_collection(
    #    collection_name="inat_iss",
    #    vectors_config=VectorParams(size=512, distance=Distance.COSINE),
    #)

    print("loading clip as st model")
    st_model = SentenceTransformer("clip-ViT-B-32")
    print("loading val ds from 2.12")
    ds = pd.read_csv("/data-ssd/alex/datasets/vision-export-20240218060015-aka-2.12/val_cleaned.csv", nrows=1_000)
    pbar = tqdm(total=len(ds))
    points=[]

    for key in logging.Logger.manager.loggerDict:
        logging.getLogger(key).disabled = True

    for i, row in ds.iterrows():
        if os.path.exists(row.filename):
            image = Image.open(row.filename)
            img_emb = st_model.encode(image).astype(np.float32).tolist()
            body = row.to_dict()
            points.append(
                PointStruct(id=i, vector=img_emb, payload=body)
            )
        else:
            print("{} doesn't exist".format(row.filename))
        pbar.update(1)

    operation_info = client.upsert(
        collection_name = "inat_iss",
        wait=True,
        points=points
    )

if __name__ == "__main__":
    main()
