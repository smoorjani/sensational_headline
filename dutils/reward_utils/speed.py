import numpy as np
from word_embedding_measures.utils.embeddings import get_chunk_embeddings
from word_embedding_measures.utils.data import get_data_chunks, get_word_tokenized_corpus
from word_embedding_measures.utils.features import get_speed

def compute_speeds(data, ft_model, stemmer, en_stop):
    documents = get_word_tokenized_corpus(data, stemmer, en_stop)

    # print('Chunking...')
    chunks = [get_data_chunks(document, chunk_len=3, mode='chunk_len') for document in documents]
    # print('Embedding...')
    chunk_embs = np.array([get_chunk_embeddings(ft_model, chunk) for chunk in chunks])
    # print('Computing Features...')
    features = [get_speed(np.stack(chunk_emb))[-1] if len(chunk_emb) > 1 else 0. for chunk_emb in chunk_embs]
    return np.array(features)