# %%
import socket
# from langchain.document_loaders import ArxivLoader
# from langchain.document_loaders import pdf
# from langchain.document_loaders.merge import MergedDataLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings, LlamaCppEmbeddings
from langchain_community.llms import llamacpp
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import utils as u

print(socket.gethostname())

# %%
DATA_ROOT = "../data/"
pdf_filenames = u.list_files(DATA_ROOT)
print("pdf_filenames", pdf_filenames)

# %%
# splitData = []
# for pdf in pdf_filenames:
#     pdfLoader = PyPDFLoader(DATA_ROOT + pdf)
#     splitData.extend(pdfLoader.load_and_split())

# # %%
# MODEL_NAME = "BAAI/bge-small-en-v1.5"
# # "google/gemma-2b"#"mistralai/Mistral-7B-Instruct-v0.2" #"microsoft/Orca-2-13b"
# # # "mistralai/Mixtral-8x7B-v0.1"

# encode_kwargs = {"normalize_embeddings": True}  # set True to compute cosine similarity

# # https://api.python.langchain.com/en/latest/embeddings/langchain_community.embeddings.huggingface.HuggingFaceBgeEmbeddings.html
# hf_bge_embeddings = HuggingFaceBgeEmbeddings(
#     model_name=MODEL_NAME,
#     model_kwargs={"device": "cuda"},
#     encode_kwargs=encode_kwargs,
# )
# # %%
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=512,
#     chunk_overlap=16,
#     length_function=len,
# )

# docs = text_splitter.split_documents(all_docs)
# %%
vectorstore = Chroma.from_documents(splitData, hf_bge_embeddings)

# %%

base_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# %%
# relevant_docs = base_retriever.get_relevant_documents(
#     "K-means take less time and memory than agglomerative \
#         hierarchical clustering and is the most efficient clustering algorithm possible."
# )

# %%
# print(relevant_docs[0].page_content)
# # %%
# query = "what is this document about ?"
# result = vectorstore.similarity_search_by_vector(hf_bge_embeddings.embed_query(query))
# print(result[0].page_content)

# %%
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])


llm = llamacpp.LlamaCpp()

# %%
llamaEmbd = LlamaCppEmbeddings(
    model_path="../Models/mistral-7b-instruct-v0.1.Q3_K_M.gguf",
    verbose=True,
    )

# %%
splitData = []
for pdf in pdf_filenames:
    pdfLoader = PyPDFLoader(DATA_ROOT + pdf)
    data = pdfLoader.load_and_split()
    for d in data:
        splitData.append(d.page_content)
#%%
splitData

#%%
  
    

#%%

#%%
embdedLLmaa = llamaEmbd.embed_documents(splitData) #embed_documents(texts=splitData)
# llm.generate("What is the document about ?")

# %%
