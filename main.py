from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vectors import retriever
model = OllamaLLM(model="llama3.2")

template = """
Assumme you are very good in answering questions about research papers

Here are the relevant research paper excerpts: {papers}

Here is the question that the user is asking: {question}

Please give your answer as if an expert were speaking!

"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

while True:
    print("\n------------")
    question=input("Ask any question relevant to the paper? (press q/Q to quit)\n")
    print("\n")
    
    if question=="q" or question=="Q":
        break

    papers = retriever(question)
    output = chain.invoke({"papers":papers, "question":question})
    print(output)

print("Thankyou, come back!")