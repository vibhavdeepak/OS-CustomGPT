from langchain.llms import GPT4All

llm = GPT4All(model='./models/ggml-gpt4all-j-v1.3-groovy.bin')

llm("What is Niggas in Paris")
llm_chain = LLMChain(prompt=prompt, llm=llm)
