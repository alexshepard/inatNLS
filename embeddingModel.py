from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)

    def get_embedding(self, input_data):
        return self.model.encode(input_data)
