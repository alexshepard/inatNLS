# testing NL search with elastic and iNat data and CLIP

1. install elasticsearch and run it. I'm using docker: https://www.elastic.co/guide/en/elasticsearch/reference/current/docker.html
2. make a python venv and activate it
3. `pip install -r requirements.txt`
4. copy `config.yml.sample` to `config.yml` and edit it
5. download a mediapipe face detector model from google: https://developers.google.com/mediapipe/solutions/vision/face_detector#models - I'm using the BlazeFace (short-range) since that's the only model available at the time of this writing.
6. `flask reindex complete_1k_obs_sample.csv`  # this will take a while, mostly due to photo downloading
7. `flask run`
8. visit localhost:5001 in your browser