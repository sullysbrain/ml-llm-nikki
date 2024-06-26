# Project Nikki: Large Language Model (LLM) App with LangChain, Streamlit and Local Transformer (such as Llama3)

A framework for a web-based LLM (AI) powered language tutor. The language model is constructed with the LangChain framework and 'deployed' locally as a locally hosted server using StreamLit. The model implements llama3:8b, but other models such as Mixtral:8x7b can be swapped out in one line of code.

Additionally, the app uses Sentence-Transformers in order to add RAG functionality for vector embedding of local documents. This is the retrieved using ChromaDB's retriever, thereby being able to have a fully functional LLM with RAG running local with no external API calls in order to protect proprietary data.

Streamlit has been implemented and the UI enhanced to be more like a standard chat interface. For this initial iteration, the chat history has been disabled as I make some refactoring tweaks for future enhancements.

TODO: Dockerize the application and run via cloud service.


## Ollama Framework
Note, download ollama to run the llama3 model locally:
- Install:  
	- ollama pull llama3:8b
	- ollama pull mixtral:8x7b

- Run:
-- ollama run llama3
-- blobs / models stored here:
	- ~/.ollama/models


## TODO:
- Dockerize the project
- Add AWS
- Add Agents and Tools (ie. for reading CSV files in real time and analyzing them)
