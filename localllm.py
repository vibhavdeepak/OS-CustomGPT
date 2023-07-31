from langchain import PromptTemplate, LLMChain
from langchain.llms import GPT4All
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

template = """
You are a friendly chatbot assistant that responds in a conversational
manner to users questions. Keep the answers short, unless specifically
asked by the user to elaborate on something. Give detailed answers.

Question: {question}

Answer:"""
prompt = PromptTemplate(template=template, input_variables=["question"])

llm = GPT4All(
    model='./models/ggml-gpt4all-j-v1.3-groovy.bin',
    callbacks=[StreamingStdOutCallbackHandler()]
)

llm_chain = LLMChain(prompt=prompt, llm=llm)

query = input("Prompt: ")
llm_chain(query)




'''
import os
from langchain import PromptTemplate, LLMChain
from langchain.llms import CerebriumAI


template = """
You are a friendly chatbot assistant that responds in a very conversational tone. If you are asked to elaborate on something, do so.
I want long answers for most prompts. Always finish your sentences. If you are asked a question, answer it.
Question: {question}

Answer:"""

prompt = PromptTemplate(template=template, input_variables=["question"])

llm_chain = LLMChain(prompt=prompt)

green = "\033[0;32m"
white = "\033[0;39m"

while True:
    query = input(f"{green}Prompt: ")
    if query == "exit" or query == "quit" or query == "q":
        print('Exiting')
        break
    if query == '':
        continue
    response = llm_chain(query)
    print(f"{white}Answer: " + response['text'])'''