# Building an AI Large Language Model (LLM) App with LangChain, Streamlit and Local Transformer (such as Llama3)

I built out a web deployed LLM built with the help of LangChain and 'deployed' locally as a locally hosted server using StreamLit. The model implements llama3:8b, but other models such as Mixtral:8x7b can be swapped out in one line of code.

Additionally, the app uses Sentence-Transformers in order to add RAG functionality for vector embedding of local documents. This is the retrieved using ChromaDB's retriever, thereby being able to have a fully functional LLM with RAG running local with no external API calls in order to protect proprietary data.

Next up is building out the UI with Streamlit.


## Ollama Framework
Note, download ollama to run the llama3 model locally:
- Install:  
	- ollama pull llama3:8b

- Run:
-- ollama run llama3
-- blobs / models stored here:
	- ~/.ollama/models


## TODO:
- Configure Streamlit for a more user friendly UI
- Add AWS
	
