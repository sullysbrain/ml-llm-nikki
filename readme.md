# Project Nikki: Large Language Model (LLM) App with LangChain, Streamlit and Local Transformer (such as Llama3)

TODO: Build up the LoRA for personality instead of doing in PromptTemplate
TODO: Dockerize the application and run via cloud service.
TODO: add user accounts


A framework for a web-based LLM (AI) powered language tutor. The language model is constructed with the LangChain framework and 'deployed' locally as a locally hosted server using StreamLit. The model implements llama3:8b, but other models such as Mixtral:8x7b can be swapped out in one line of code.

Additionally, the app uses Sentence-Transformers in order to add RAG functionality for vector embedding of local documents. This is the retrieved using ChromaDB's retriever, thereby being able to have a fully functional LLM with RAG running local with no external API calls in order to protect proprietary data.

Streamlit has been implemented and the UI enhanced to be more like a standard chat interface. For this initial iteration, the chat history has been disabled as I make some refactoring tweaks for future enhancements.

Added "writer assistant" personality to begin addiitonal functionality. 


## Ollama Framework
Note, download ollama to run the llama3 model locally:
- Install:  
	- ollama pull llama3:8b
	- ollama pull mixtral:8x7b

- Run:
-- ollama run llama3
-- blobs / models stored here:
	- ~/.ollama/models

Current 'best' model for tutor app is mixtral. Updated as new ones are released.


## CRON DB UPDATE
I setup a crontab job to run each day to scrape for more data added to the documents folder, and then build a new ChromaDB embedded database from that. Specifically, I run "embed_docs.py" every day at 8am.

To do this, type 'crontab -e' to edit the cron table. Then add this line:

	0 8 * * * ./python embed_docs.py



# PDF Creation
Add weasyprint for markdown -> css -> pdf creation.
	brew install weasyprint





## TODO:
- Add Users to store user-based history
- Add voice
- Dockerize the project
- Add AWS
- Add Agents and Tools (ie. for reading CSV files in real time and analyzing them)
