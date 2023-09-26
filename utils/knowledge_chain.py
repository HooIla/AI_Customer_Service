import os
import logging
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain



def chain(llm, retriever, prompt):
    logging.info('Initializing RetrievalQA Chain...')
    knowledge_chain = RetrievalQA.from_llm(
        llm = llm,
        retriever = retriever,
        prompt = prompt,
        return_source_documents = True
    )
    knowledge_chain.combine_documents_chain.document_prompt = PromptTemplate(
        input_variables=["page_content"],
        template="{page_content}"
    )
    return knowledge_chain


def memory_chain(llm,prompt,chain_type,retriever,memory):
    print('Initializing ConversationalRetrievalChain Chain...')
    memory_chain = ConversationalRetrievalChain.from_llm(llm,
                                        condense_question_prompt=prompt,
                                        chain_type=chain_type, 
                                        retriever=retriever,
                                        memory=memory)
    return memory_chain
