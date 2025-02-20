import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Not currently used
def euclidian_distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def sort_by_nearest(client, embedding_model, key_text, query_texts):
    '''
    Use the supplied OpenAI client to compare the embeddings of the query_texts to
    the embedding of the key_text, then return the query_text with the most similar
    embedding to the key_text using cosine similarity.

    client (OpenAI client) -- client makes the request
    embedding_model (str) -- a valid OpenAI API model string, e.g. 'text-embedding-3-large'
    key_text (str) -- a string which will be embedded and used as a reference for the query_texts. 
    query_texts (list[str]) -- a list of strings which will be embedded and compared to the key_text.
    '''
    key_embedding = client.embeddings.create(input = [key_text], model=embedding_model).data[0].embedding

    for_sorting = []
    for query_text in query_texts:
        query_embedding = client.embeddings.create(input = [query_text], model=embedding_model).data[0].embedding
        for_sorting.append((query_text, cosine_similarity(key_embedding, query_embedding)))

    # sort them by the computed cosine similarity
    for_sorting.sort(key = lambda x: x[1], reverse=True)

    return for_sorting
