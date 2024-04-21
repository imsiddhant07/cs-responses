# from sentence_transformers import SentenceTransformer
# model = SentenceTransformer('BAAI/bge-base-en-v1.5')


def compute_similarity(model, source, target):
    """
    Calculate the cosine similarity between embeddings of source and target strings.

    This function utilizes a pre-trained model to encode the source and target strings into embeddings,
    normalizes these embeddings, and computes their dot product to measure cosine similarity.

    Args:
        - model (SentenceTransformer): A pre-trained SentenceTransformer model capable of encoding text into embeddings.
        - source (str): The source text string from which to generate the embedding.
        - target (str): The target text string from which to generate the embedding.

    Returns:
        - similarity (float) : The cosine similarity score between the source and target embeddings.

    Note:
    The function assumes that the `model` passed has an `encode` method which supports the `normalize_embeddings` parameter.
    The cosine similarity is a value between 0 and 1, where 1 indicates perfect similarity, 0 indicates no similarity.
    """
    source_embeddings = model.encode(source, normalize_embeddings=True)
    target_embeddings = model.encode(target, normalize_embeddings=True)
    similarity = source_embeddings @ target_embeddings.T
    return similarity
