from langchain_ollama import ChatOllama


model = ChatOllama(
    model="llama3.2",
    temperature=0,
)

for chunk in model.stream("Write me a 1 verse song about goldfish on the moon"):
    print(chunk.content, end="|", flush=True)
