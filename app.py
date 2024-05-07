import logging

import click
from flask import Flask, render_template, request

from config import Config
from search import Search
from requestFormatter import RequestFormatter

logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s"
)


def create_app():
    app = Flask(__name__)
    app_config = Config.load_config()
    app.config.update(app_config)

    search = Search(
        app.config["CLIP_MODEL_NAME"],
        app.config["IMAGE_CACHE_DIR"],
        app.config["INSERT_BATCH_SIZE"],
    )
    app.search = search

    handler = logging.StreamHandler()
    rf = RequestFormatter(
        "[%(asctime)s] %(remote_addr)s requested %(url)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(rf)
    app.logger.addHandler(handler)

    return app


app = create_app()


iconic_taxa = {
    "48460": "Life",
    "48460/1/2/355675/20978": "Amphibians",
    "48460/1": "Animals",
    "48460/1/47120/245097/47119": "Arachnids",
    "48460/1/2/355675/3": "Birds",
    "48460/48222": "Chromistans",
    "48460/47170": "Fungi",
    "48460/1/47120/372739/47158": "Insects",
    "48460/1/2/355675/40151": "Mammals",
    "48460/1/47115": "Molluscs",
    "48460/47126": "Plants",
    "48460/47686": "Protozoans",
    "48460/1/2/355675/47178": "Ray-finned Fishes",
    "48460/1/2/355675/26036": "Reptiles",
}

continent_choices = [
    "Worldwide",
    "Africa",
    "Asia",
    "Europe",
    "North America",
    "Oceania",
    "South America",
]


@app.get("/")
def index():
    return render_template(
        "index.html",
        continent_choices=continent_choices,
        iconic_taxa=iconic_taxa,
    )


@app.post("/")
def handle_search():
    query = request.form.get("query", "")
    query_vector = app.search.get_embedding(query)

    login = request.form.get("login", "")
    taxon_name = request.form.get("taxon_name", "")
    continent = request.form.get("continent", "")
    iconic_taxon = request.form.get("iconic_taxon", "")

    filters = {"filter": []}
    if taxon_name != "":
        filters["filter"].append(
            {"term": {"name.keyword": {"value": taxon_name}}})
    if login != "":
        filters["filter"].append(
            {"term": {"observer_login.keyword": {"value": login}}})
    if continent != "" and continent != "Worldwide":
        filters["filter"].append(
            {"term": {"continent.keyword": {"value": continent}}})
    if iconic_taxon != "" and iconic_taxon != "None":
        # this is super inefficient but should be fine for a prototype
        filters["filter"].append(
            {"prefix": {"ancestry.keyword": {"value": iconic_taxon}}}
        )

    results = app.search.search(
        index_name=app.config["ES_INDEX_NAME"],
        knn={
            "field": "embedding",
            "query_vector": query_vector,
            "k": app.config["KNN"]["K"],
            "num_candidates": app.config["KNN"]["NUM_CANDIDATES"],
            **filters,
        },
        size=app.config["KNN"]["K"],
        from_=0,
    )

    return render_template(
        "index.html",
        query=query,
        login=login,
        taxon_name=taxon_name,
        continent=continent,
        continent_choices=continent_choices,
        iconic_taxon=iconic_taxon,
        iconic_taxa=iconic_taxa,
        results=results["hits"]["hits"],
        from_=0,
        total=results["hits"]["total"]["value"],
    )


@app.cli.command()
@click.argument("filename", required=True)
def reindex(filename):
    """Add new data to elasticsearch index."""
    # because we can't upsert into elastic search
    # we'll create duplicates if we're not careful
    # so instead we just recreate the index every time
    # this is fine for a demo/prototype
    app.search.delete_index(app.config["ES_INDEX_NAME"])
    app.search.create_index(app.config["ES_INDEX_NAME"])
    app.search.add_to_index(app.config["ES_INDEX_NAME"], filename)
