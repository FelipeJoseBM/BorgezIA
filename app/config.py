from langchain_community.embeddings import HuggingFaceBgeEmbeddings

# Nome do modelo de embeddings
MODEL_NAME = "BAAI/bge-small-en"

def get_embedding_function():
    """
    Retorna a função de embeddings configurada com o modelo definido.
    """
    return HuggingFaceBgeEmbeddings(
        model_name=MODEL_NAME,
        encode_kwargs={'normalize_embeddings': True}
    )