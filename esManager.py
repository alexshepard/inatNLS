import logging

from elasticsearch import Elasticsearch

logger = logging.getLogger(__name__)
logger.level = logging.DEBUG


class ElasticSearchManager:
    def __init__(self, host="localhost", port=9200):
        self.es = Elasticsearch("http://localhost:9200")
        # self.es = Elasticsearch([{'host': host, 'port': port}])
        logger.info("Connected to Elasticsearch!")

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

    def search(self, index_name, **query_args):
        return self.es.search(index=index_name, **query_args)

    def index_item(self, index_name, document):
        return self.es.index(index=index_name, document=document)

    def bulk_insert(self, operations):
        return self.es.bulk(operations=operations)
