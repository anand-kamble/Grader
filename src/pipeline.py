

# %%############## IMPORTS ################
from langchain.llms import llamacpp
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


# %%############## LLM ################
llm = llamacpp.LlamaCpp(model_path=MODEL_DIR + MODEL_NAME)

# %%############## PROMPT ################
template = """
You are a professor of graduate level course data mining.
The question asked is: `{question}`
rate the following answer on a scale of 1 to 10, where 1 is the worst and 10 is the best:
`{answer}`
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["answer","question"],
)

# %%############## LLM CHECK ################
llm(
    prompt.format(
        answer="K-means is a clustering algorithm that divides the data into K clusters",
        question="What is K-means?"
    )
)


# %%############## CHAIN 1 ################
first_chain = LLMChain(
    llm=llm,
    prompt=prompt,
)
