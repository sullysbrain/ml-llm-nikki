# import ollama
# response = ollama.chat(model='llama2', messages=[
#   {
#     'role': 'user',
#     'content': 'Why is the sky blue?',
#   },
# ])
# print(response['message']['content'])


from langchain_community.llms import Ollama
llm = Ollama(model="llama2")

from langchain_core.prompts import ChatPromptTemplate


file_path = 'report.txt'
file = open(file_path, 'r')
content = file.read()
file.close()
with open('_private/report.txt', 'r') as file:
    content = file.read()

while True:
    
    userquestion = input("Enter your question: ")
    system_prompt = "Here is some background information: " + content + "  Keep your answers to about one or two sentences."

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", userquestion)
        ])
    chain = prompt | llm 

    print(chain.invoke({"human": userquestion}))
    print("\n\n")


