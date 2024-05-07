import logging

import click
from flask import Flask, render_template, request

from config import Config
from data import iconic_taxa, continent_choices
from requestFormatter import RequestFormatter
from search import Search, SearchService


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

    searchService = SearchService(
        app.config, search
    )
    app.searchService = searchService

    handler = logging.StreamHandler()
    rf = RequestFormatter(
        "[%(asctime)s] %(remote_addr)s requested %(url)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(rf)
    app.logger.addHandler(handler)

    return app


app = create_app()


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

    results = app.searchService.perform_search(
        query, login, taxon_name, continent, iconic_taxon
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
