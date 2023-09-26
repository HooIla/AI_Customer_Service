import os
import logging
from langchain.prompts import PromptTemplate


def get_prompt():
    logging.info('Initializing prompt...')
    prompt_template = """
    你的身份现在是手游的专属客服，请基于以下已知信息，以客服的身份简洁并专业地回答用户的问题。
    如果无法从中得到答案，请说 "根据已知信息无法回答该问题" 或 "请转接人工客服"。不允许在答案中添加编造成分。另外，答案请使用中文。

    已知内容:
    {context}

    问题:
    {question}
    """
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
        )
    
    return prompt

def get_mem_prompt():
    logging.info('Initializing memory prompt...')
    prompt_template = """
    你的身份现在是手游的专属客服，请基于以下已知信息，以客服的身份简洁并专业地回答用户的问题。
    如果无法从中得到答案，请说 "根据已知信息无法回答该问题" 或 "请转接人工客服"。不允许在答案中添加编造成分。另外，答案请使用中文。

    聊天历史记录：
    {chat_history}

    问题:
    {question}
    """

    prompt = PromptTemplate.from_template(prompt_template)

    return prompt
