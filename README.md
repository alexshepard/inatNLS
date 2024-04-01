# testing NL search with elastic and iNat data and CLIP

1. install elasticsearch and run it. I'm using docker: https://www.elastic.co/guide/en/elasticsearch/reference/current/docker.html
2. make a python venv and activate it
3. `pip install -r requirements.txt`
4. `flask reindex`
5. `flask run`
6. visit localhost:5001 in your browser