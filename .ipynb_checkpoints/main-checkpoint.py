# %%
import socket

print(socket.gethostname())

# pip install -U langchain openai ragas arxiv pymupdf chromadb tiktoken accelerate bitsandbytes datasets sentence_transformers FlagEmbedding ninja  tqdm rank_bm25 transformers
# flash_attn --no-build-isolation
# %%
import utils as u


# from langchain.document_loaders import ArxivLoader
# from langchain.document_loaders import pdf
from langchain.document_loaders.merge import MergedDataLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

# papers = ["2310.13800", "2307.03109", "2304.08637", "2310.05657", "2305.13091", "2311.09476", "2308.10633", "2309.01431", "2311.04348"]

# docs_to_merge = []
# %%
DATA_ROOT = "data/"
pdf_filenames = u.list_files(DATA_ROOT)
print("pdf_filenames", pdf_filenames)

# %%
splitData = list()
for pdf in pdf_filenames:
    pdfLoader = PyPDFLoader(DATA_ROOT + pdf)
    splitData.extend(pdfLoader.load_and_split())
    
 #%%
import getpass

inference_api_key = getpass.getpass("Enter your HF Inference API Key:\n\n")

print(inference_api_key) 

    
# %%
model_name = "google/gemma-2b"#"mistralai/Mistral-7B-Instruct-v0.2" #"microsoft/Orca-2-13b"  # "mistralai/Mixtral-8x7B-v0.1"

encode_kwargs = {"normalize_embeddings": True}  # set True to compute cosine similarity

# https://api.python.langchain.com/en/latest/embeddings/langchain_community.embeddings.huggingface.HuggingFaceBgeEmbeddings.html
hf_bge_embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name, 
    model_kwargs={"device": "cuda"}, 
    encode_kwargs=encode_kwargs,
    api_key="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"
    
)
# %%
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512, chunk_overlap=16, length_function=len, 
)

# docs = text_splitter.split_documents(all_docs)
# %%
vectorstore = Chroma.from_documents(splitData, hf_bge_embeddings)

# %%

base_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

#%%
relevant_docs = base_retriever.get_relevant_documents(
    "What are the challenges in evaluating Retrieval Augmented Generation pipelines?"
)