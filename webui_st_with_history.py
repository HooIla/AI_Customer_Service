import os
import streamlit as st
from configs.configs import *
from utils.load_prompt import get_mem_prompt
from utils.knowledge_chain import memory_chain
from utils.load_data import upload_csv, init_data
from utils.load_model import init_embedding_model, init_llm_model
from langchain.memory import ConversationBufferWindowMemory



def init_all():
    hf_embeddings = init_embedding_model(EMBEDDING_MODEL_NAME, EMBEDDING_MODEL_PATH)
    llm = init_llm_model(MODEL_NAME, API_KEY, API_BASE, TEMPERATURE)
    prompt = get_mem_prompt()

    return hf_embeddings, llm, prompt

@st.cache_resource
def init_mem(_llm,_prompt,_retriever):
    memory = ConversationBufferWindowMemory(k=3, llm=_llm, memory_key="chat_history",return_messages=True)
    mem_chain = memory_chain(
        llm = _llm,
        prompt = _prompt,
        chain_type = "stuff",
        retriever = _retriever,
        memory = memory
    )

    return memory, mem_chain

def init_chat_history(name):
    with st.chat_message("assistant", avatar='assistant'):
        st.markdown(f"您好，我是《{name}》的专属智能客服，很高兴为您服务")
    if "messages" in st.session_state:
        for message in st.session_state.messages:
            avatar = 'user' if message["role"] == "user" else 'assistant'
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])
    else:
        st.session_state.messages = []
    return st.session_state.messages

def clear_history(memory):
    memory.clear()


st.set_page_config(page_title='智能客服')
st.sidebar.markdown("# 知识库")


hf_embeddings, llm, prompt = init_all()

# Sidebar
new_db_name = st.sidebar.text_input("新建知识库","")
if new_db_name:
    upload_file = st.sidebar.file_uploader("upload", type="csv")
    if upload_file:
        upload_csv(upload_file, hf_embeddings, new_db_name)
        st.toast("知识库-{new_db_name}创建成功！")

knowledge_list = [name for name in os.listdir(KNOWLEDGE_PATH) if os.path.isdir(os.path.join(KNOWLEDGE_PATH, name))]
db_select = st.sidebar.selectbox("知识库选择", knowledge_list)

retriever = init_data(hf_embeddings, db_select)






# Chat
if retriever:
    messages = init_chat_history(db_select)
    memory, mem_chain = init_mem(llm,prompt,retriever)

    if st.sidebar.button("clear history"):
        clear_history(memory)
    
    if query := st.chat_input():
        with st.chat_message("user", avatar="user"):
            st.markdown(query)
        response = mem_chain(query)
        messages.append({"role": "user", "content": query})
        
        

        with st.chat_message("assistant", avatar="assistant"):
            placeholder=st.empty()
            placeholder.markdown(response['answer'])
        messages.append({"role": "assistant", "content": response['answer']})
        print(response)  




