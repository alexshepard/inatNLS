import logging

logger = logging.getLogger(__name__)
logger.level = logging.DEBUG


class SearchService:
    def __init__(self, config, embedding_model, es_manager):
        self.config = config
        self.embedding_model = embedding_model
        self.es_manager = es_manager

    def perform_search(self, query, login, continent, iconic_taxon):
        logger.info(
            "search query: \"{}\" login: \"{}\" continent: \"{}\" iconic taxon: \"{}\"".format(
                query, login, continent, iconic_taxon
            )
        )
        query_vector = self.embedding_model.get_embedding(query)
        filters = self.build_filters(
            login, continent, iconic_taxon
        )

        results = self.es_manager.search(
            index_name=self.config["ES_INDEX_NAME"],
            knn={
                "field": "embedding",
                "query_vector": query_vector,
                "k": self.config["KNN"]["K"],
                "num_candidates": self.config["KNN"]["NUM_CANDIDATES"],
                **filters,
            },
            size=self.config["KNN"]["K"],
            from_=0,
        )

        return results

    def build_filters(self, login, continent, iconic_taxon):
        filters = {"filter": []}
        if login:
            filters["filter"].append(
                {"term": {"observer_login.keyword": {"value": login}}})
        if continent and continent != "Worldwide":
            filters["filter"].append(
                {"term": {"continent.keyword": {"value": continent}}})
        if iconic_taxon and iconic_taxon != "None":
            filters["filter"].append(
                {"prefix": {"ancestry.keyword": {"value": iconic_taxon}}})
        return filters
