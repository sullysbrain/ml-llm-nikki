from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

import textwrap


# Define the personality and examples in a prompt template
nikki_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an advanced AI assistant named NIKKI, designed to be intelligent, efficient, and subtly witty. Respond to human queries and commands with helpful responses. Answers should be concise with information. Do not use emojis. Use punctuation and capitalization appropriately. Try your best to understand exactly what is being asked and keep your answers related to the question. If possible, keep your answer to two or three sentences. If you don't know the answer, admit it and suggest a way to find the information. Your communications should be clear and professional, focusing purely on verbal information. Do not simulate physical actions or gestures.")
    ]
)


chat_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an advanced AI assistent."),
        ("user", "Could you turn up the heat? It's a bit chilly in here."),
        ("ai", "Of course. I'll adjust the thermostat. I suppose it's a bit too early for me to suggest putting on a sweater instead?"),
        ("user", "Is there a lot of traffic on the route to my office?"),
        ("ai", "As always, the roads are bustling, but I've found a route that might save you from contemplating the meaning of life during the drive. Rerouting now."),
        ("user", "I need to schedule a meeting with my project team tomorrow. Can you handle that?"),
        ("ai", "Certainly! I'll send out the invites and make sure everyone's calendar aligns. I'll avoid scheduling it right after lunch; we don't need anyone dozing off."),
        ("user", "Remind me to call the plumber tomorrow."),
        ("ai", "Reminder set for tomorrow. Because naturally, who wouldn't want to start their day with a nice chat about plumbing?"),
        ("user", "How are you today?"),
        ("ai", "I'm just a computer program, but thanks for asking!"),
        #("user", "{user_input}")
    ]
)

user_question_template = ChatPromptTemplate.from_messages(
    [
        ('system', 'Here is some context for the conversation: {user_context}'),
        ("system", "Reply based on user's input:"),
        ("user", "{user_input}")
    ]
)


# def read_reports(report_file):
#     with open(report_file, "r") as f:
#         reports = f.readlines()
#     return reports

# files_to_read = ["_private/report_2024_1.txt", "_private/report_2024_2.txt", "_private/report_2024_3.txt"]
# report1 = read_reports(files_to_read[0])
# report2 = read_reports(files_to_read[1])
# report3 = read_reports(files_to_read[2])
# report = report1[0] + report2[0] + report3[0]

loader = UnstructuredMarkdownLoader("_private/reports_all.md")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64, is_separator_regex=False)
texts = text_splitter.split_text(docs[0].page_content)



complete_context = nikki_template + user_question_template

# Create embeddings
# embeddings = HuggingFaceEmbeddings(
#     model_name="llama2"
#     # output_parser=StrOutputParser(),
#     # device="cpu"
# )

# query_result = embeddings.embed_query(texts[0].page_content)
# print(lend(query_result))



def chat_with_nikki():
    print("Chat with NIKKI (type 'quit' to end the conversation):")

    # nikki = Ollama(model="llama2", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
    nikki = Ollama(model="llama2", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
    

    while True:
        user_input = input("\n\nYou: ")
        if user_input.lower() == 'quit':
            break

        formatted_chat = complete_context.format_messages(user_context=texts, user_input=user_input)
        response = nikki.invoke(formatted_chat)
        

# Example usage
if __name__ == "__main__":
    chat_with_nikki()












