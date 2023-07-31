import os
from langchain import PromptTemplate, LLMChain
from langchain.llms import CerebriumAI

os.environ["CEREBRIUMAI_API_KEY"] = "public-61b96b7a04921bccf00c"

template = """
You are a friendly chatbot assistant that responds in a very conversational tone. If you are asked to elaborate on something, do so.
I want long answers for most prompts. Always finish your sentences. If you are asked a question, answer it.
Question: {question}

Answer:"""

prompt = PromptTemplate(template=template, input_variables=["question"])
llm = CerebriumAI(
  endpoint_url="https://run.cerebrium.ai/gpt4-all-webhook/predict",
  max_length=200
)
llm_chain = LLMChain(prompt=prompt, llm=llm)

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
    print(f"{white}Answer: " + response['text'])