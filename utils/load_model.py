import os
import logging
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings

# logging.basicConfig(filename=os.path.join(LOG_DIR, 'app.log'), level=logging.INFO)


# Huggingface embedding model with local path
def init_embedding_model(model_name, cache_folder):
    logging.info('Initializing embedding model...')
    hf_embeddings = HuggingFaceEmbeddings(
        model_name = cache_folder,
        model_kwargs = {'device': 'cpu'},
        cache_folder = cache_folder
    )
    return hf_embeddings

def init_llm_model(model_name, openai_api_key, openai_api_base, temperature):
    logging.info('Initializing llm model...')
    llm = ChatOpenAI(
        model_name = model_name,
        openai_api_key = openai_api_key,
        openai_api_base = openai_api_base,
        temperature = temperature
    )
    return llm
