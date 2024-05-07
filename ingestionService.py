import csv
import logging
import os

logger = logging.getLogger(__name__)
logger.level = logging.DEBUG


class IngestionService:
    def __init__(self, image_manager, es_manager, embedding_model):
        self.image_manager = image_manager
        self.es_manager = es_manager
        self.embedding_model = embedding_model

    def ingest_data(self, data_file, index_name):
        # because we can't upsert into elastic search
        # we'll create duplicates if we're not careful
        # so instead we just recreate the index every time
        # this is fine for a demo/prototype
        self.es_manager.delete_index(index_name)
        self.es_manager.create_index(index_name)

        with open(data_file, "r") as file:
            num_docs = len(file.readlines()) - 1

        with open(data_file) as csvfile:
            csvreader = csv.DictReader(csvfile)
            seen = 0
            for row in csvreader:
                seen += 1

                local_path = self.image_manager.path_for_photo_id(
                    row["photo_id"]
                )
                photo_url = self.image_manager.url_for_photo_id(
                    row["photo_id"], row["extension"]
                )

                if not os.path.exists(local_path):
                    if not self.image_manager.download_image(photo_url, local_path):
                        logger.warn("can't download {}, skipping {}".format(
                            photo_url, local_path
                        ))
                        continue

                try:
                    img = self.image_manager.open_image(local_path)
                    img_emb = self.embedding_model.get_embedding(img)
                    document = {
                        **row,
                        "embedding": img_emb,
                    }
                    self.es_manager.index_item(
                        index_name=index_name, document=document
                    )

                    if seen % 100 == 0:
                        logger.info(
                            "ingestion job {} / {}".format(seen, num_docs)
                        )
                except Exception as e:
                    logger.error(
                        "couldn't open or encode {}: {}".format(
                            local_path, repr(e)
                        )
                    )
                    continue
