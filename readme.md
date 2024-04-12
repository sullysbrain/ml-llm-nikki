## Building an AI Large Language Model (LLM) App with LangChain, Streamlit, OpenAI, and Llama2

I built out a web deployed LLM built with the help of LangChain and 'deployed' locally as a locally hosted server using StreamLit. The model itself currently uses the OpenAI API but my intent is to shift over to Meta's and Microsoft's Llama 2.0.

As mentioned above, the web interface uses StreamLit running as localhost on my machine. Eventually, I plan to roll out using AWS.


# AWS CLI Commands

## Get list of EC2 Instances
	aws ec2 describe-instances > saved_file.json
		* this will save the instances as a locally saved .json file

	aws ec2 start-instances --instance-ids <<instance-id>>
	aws ec2 stop-instances --instance-ids <<instance-id>>



## Ollama Framework
Note, download ollama to run the llama2 model locally:
- Install:  
	- 7B model (3.8 GB):  ollama pull llama2
	- 13B model (7.3 GB):  ollama pull llama2:13b

- Run:
-- ollama run llama2
-- blobs / models stored here:
	- ~/.ollama/models
