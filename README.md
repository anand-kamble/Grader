## Installed packages using
`pip install -U langchain openai ragas arxiv pymupdf chromadb tiktoken accelerate bitsandbytes datasets sentence_transformers FlagEmbedding ninja  tqdm rank_bm25 transformers`


### Models
`mistralai/Mixtral-8x7B-v0.1`
This model too large for my system, and that's why I havn't tested it yet. 

Right now, using `microsoft/Orca-2-13b` since it can be ran on the classroom machine.
This model ran out of VRAM.

## Execution
I recommend running the `main.py` file using VSCode interactive mode. It will make it easy to work with the code since ,it avoids reloading the model every time.

Alternatively, you can always use:
`python main.py`