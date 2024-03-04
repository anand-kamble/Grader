<!-- ## Installed packages using
`pip install -U langchain openai ragas arxiv pymupdf chromadb tiktoken accelerate bitsandbytes datasets sentence_transformers FlagEmbedding ninja  tqdm rank_bm25 transformers` -->
# Auto Grader

## Create a conda env using the yaml file.
`conda env create environment.yml`

## VSCode extension
I would recommend installing the following extension for linting.
This will make sure that our code styling stays same, and will avoid unnecessary changes to be recorded on git. 

Feel free to change the pylintrc as you want.
[Pylint : https://marketplace.visualstudio.com/items?itemName=ms-python.pylint](https://marketplace.visualstudio.com/items?itemName=ms-python.pylint)

### Models
`mistralai/Mixtral-8x7B-v0.1`
This model too large for my system, and that's why I havn't tested it yet. 

Right now, using `microsoft/Orca-2-13b` since it can be ran on the classroom machine.
This model ran out of VRAM.

## Execution
I recommend running the `main.py` file using VSCode interactive mode. It will make it easy to work with the code since ,it avoids reloading the model every time.

Alternatively, you can always use:
`python main.py`

In case you get an error like:
```bash
ImportError: Could not import sentence_transformers python package. Please install it with `pip install sentence_transformers`.
```

Install the `sentence_transformers` package using `pip install sentence_transformers` or any other package manager you prefer. 

https://medium.com/getpieces/how-to-build-a-langchain-pdf-chatbot-b407fcd750b9