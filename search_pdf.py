from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings


file_path = "nke-10k-2023.pdf"
loader = PyPDFLoader(file_path)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True,
)
# split the documents into chunks, each chunk is a Document object
# i.e from langchain_core.documents import Document
# Document(
#     page_content="Dogs are great companions, known for their loyalty and friendliness.",
#     metadata={"source": "mammal-pets-doc"},
# ),
all_splits = text_splitter.split_documents(docs)
for split in all_splits[:1]:
    print(split)

embeddings = OllamaEmbeddings(model="nomic-embed-text")
vector_1 = embeddings.embed_query(all_splits[0].page_content)
vector_2 = embeddings.embed_query(all_splits[1].page_content)

assert len(vector_1) == len(vector_2)
print(f"Generated vectors of length {len(vector_1)}\n")
print(vector_1[:10])
