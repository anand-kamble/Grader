# %%
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import LlamaCppEmbeddings
from langchain_community.llms import llamacpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import utils as u

# %%
DATA_ROOT = "../data/"
pdf_filenames = u.list_files(DATA_ROOT)
print("pdf_filenames", pdf_filenames)

# %%
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])


llm = llamacpp.LlamaCpp()

# %%
llamaEmbd = LlamaCppEmbeddings(
    model_path=u.Models.MISTRAL.value,
    verbose=True,
)

# %%
splitData = []
for pdf in pdf_filenames:
    pdfLoader = PyPDFLoader(DATA_ROOT + pdf)
    data = pdfLoader.load_and_split()
    for d in data:
        splitData.append(d.page_content)

# %%
embdedLLmaa = llamaEmbd.embed_documents(splitData)  # embed_documents(texts=splitData)
# llm.generate("What is the document about ?")

# %%
