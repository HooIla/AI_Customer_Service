## 1. Install 

```
python==3.10
chromadb==0.4.10
langchain==0.0.299
openai==0.28.0
streamlit==1.26.0
```



## 2. Configs

```
# 1. Change LLM api

# OpenAI
Change MODEL_NAME, API_KEY, API_BASE in configs/configs.py

# Custom llm api
Deploy your local llm with OpenAI API, then change MODEL_NAME, API_KEY, API_BASE in configs/configs.py

# 2. Change embedding Model
Change EMBEDDING_MODEL_NAME, EMBEDDING_MODEL_PATH in configs/configs.py

# 3. Change path of knowledge database
Change KNOWLEDGE_PATH in configs/configs.py
```



## 3. Webui

```
streamlit run webui_st.py		# QA with knowledge base
```

or 

```
streamlit run webui_st_with_history.py		# QA with knowledge base (3 shot history)
```



