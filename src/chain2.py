# %%
from langchain_community.document_loaders import PyPDFLoader


# %%
DATA_ROOT = "../data/"
pdf_filenames = u.list_files(DATA_ROOT)
print("pdf_filenames", pdf_filenames)
