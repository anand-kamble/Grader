# %%############## IMPORTS ################
from langchain.llms import llamacpp
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import LlamaCppEmbeddings
from utils import Models, list_files

# %%############## LLM ################
llm = llamacpp.LlamaCpp(model_path=Models.MISTRAL.value)

# %%############## DATA FROM PDF ################
DATA_ROOT = "../data/"
pdf_filenames = list_files(DATA_ROOT)
print("pdf_filenames", pdf_filenames)

# %%############## SPLIT THE DATA ################
splitData = []
for pdf in pdf_filenames:
    pdfLoader = PyPDFLoader(DATA_ROOT + pdf)
    data = pdfLoader.load_and_split()
    for d in data:
        splitData.append(d)

# %%############## EMBEDDING ################
llamaEmed = LlamaCppEmbeddings(seed=100, model_path=Models.MISTRAL.value)


# %%############## PROMPT ################
template = """
You are a professor of graduate level course data mining.
The question asked is: `{question}`
context:{context}
rate the following answer on a scale of 1 to 10, where 1 is the worst and 10 is the best:
`{answer}`
Give answer in format: Rating = x/10
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["answer", "question"],
)

# %%############## MAKING THE CONTEXT STRING ################
text_splitter = RecursiveCharacterTextSplitter()
docs = text_splitter.split_documents(splitData)
vec_store = Chroma.from_documents(splitData, llamaEmed)
base_retriever = vec_store.as_retriever(kwargs={"k": 5})

relevant_part = base_retriever.get_relevant_documents("What is k-means?")
context_filter = [
    "Introduction to Data Mining",
    "2nd Edition",
    "Tan, Steinbach, Karpatne, Kumar" "3/24/2021",
]

context_str = ""

for r in relevant_part:
    for f in context_filter:
        r.page_content = r.page_content.replace(f, "")
    context_str = context_str + r.page_content


# %%############## LLM ################
llm(
    prompt.format(
        answer="K-means is a clustering algorithm that divides the data into K clusters",
        question="What is K-means?",
        context=context_str,
    )
)


# %%############## CHAIN 1 ################
first_chain = LLMChain(
    llm=llm,
    prompt=prompt,
)

# %%
