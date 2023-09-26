import os
import logging
import tempfile
from configs.configs import *
from langchain.vectorstores import Chroma
from langchain.document_loaders.csv_loader import CSVLoader

# save csv file if upload in knowledge base
def upload_csv(upload_file, hf_embeddings,save_name):
    logging.info(f'Saving new data-{save_name}...')
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(upload_file.getvalue())
        tmp_file_path = tmp_file.name
    loader = CSVLoader(
        file_path=tmp_file_path, 
        encoding="utf-8", 
        csv_args={'delimiter': ','}
        )
    data = loader.load()
    vectorstore = Chroma.from_documents(data, hf_embeddings, persist_directory=f"{KNOWLEDGE_PATH}/{save_name}")
    vectorstore.persist()


def init_data(hf_embeddings, data_name):
    logging.info(f'Loading data-{data_name}...')
    vectorstore = Chroma(persist_directory=f"{KNOWLEDGE_PATH}/{data_name}", embedding_function=hf_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever
