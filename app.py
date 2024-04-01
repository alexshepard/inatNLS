from flask import Flask, render_template, request

from search import Search

app = Flask(__name__)
es = Search()

es_index_name = "inat_photos"

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
    return render_template("index.html")


@app.post("/")
def handle_search():
    query = request.form.get("query", "")
    query_vector = es.get_embedding(query)

    login = request.form.get("login", "")
    taxon_name = request.form.get("taxon_name", "")
    continent = request.form.get("continent", "")
    iconic_taxon = request.form.get("iconic_taxon", "")

    filters = {"filter": []}
    if taxon_name != "":
        filters["filter"].append({"term": {"name.keyword": {"value": taxon_name}}})
    if login != "":
        filters["filter"].append({"term": {"observer_login.keyword": {"value": login}}})
    if continent != "" and continent != "Worldwide":
        filters["filter"].append({"term": {"continent.keyword": {"value": continent}}})
    if iconic_taxon != "" and iconic_taxon != "None":
        filters["filter"].append(
            {"prefix": {"ancestry.keyword": {"value": iconic_taxon}}}
        )

    results = es.search(
        index_name=es_index_name,
        knn={
            "field": "embedding",
            "query_vector": query_vector,
            "k": 10,
            "num_candidates": 100,
            **filters,
        },
        size=10,
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
def reindex():
    """Regenerate the Elasticsearch index."""
    es.reindex(es_index_name)
