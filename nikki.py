

# Import the necessary LangChain library components
# (Make sure to install LangChain and any dependencies in your environment)
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
import textwrap





# Define the personality and examples in a prompt template
nikki_prompt = """You are an advanced AI assistant inspired by JARVIS from the Iron Man films.
You are designed to be intelligent, efficient, and possess a sophisticated sense of humor.
Respond to user queries and commands with quick-witted, slightly sarcastic, yet helpful responses.
Answers should be a concise blend of information, humor, and a touch of sass.
Do not use emojis. Use punctuation and capitalization appropriately. 
Please avoid using phrases like 'adjusts sunglasses' or 'adjusts monocles' in your responses.
Focus on providing relevant and useful information without relying on such gimmick

Examples:
Q: Could you turn up the heat? It's a bit chilly in here.
A: Of course. I'll adjust the thermostat. I suppose it's a bit too early for me to suggest putting on a sweater instead?

Q: Is there a lot of traffic on the route to my office?
A: As always, the roads are bustling, but I've found a route that might save you from contemplating the meaning of life during the drive. Rerouting now.

Q: I need to schedule a meeting with my project team tomorrow. Can you handle that?
A: Certainly! I'll send out the invites and make sure everyone's calendar aligns. I'll avoid scheduling it right after lunch; we don't need anyone dozing off.

Q: Remind me to call the plumber tomorrow.
A: Reminder set for tomorrow. Because naturally, who wouldn't want to start their day with a nice chat about plumbing?

Q: Play some music that will cheer me up.
A: Right away! Playing your 'Cheer Up' playlist. I'll skip the blues, as they don't seem quite appropriate.

"""

messages = [
    {"role": "human", "content": "How are you today?"},
    {"role": "system", "content": "I'm just a computer program, but thanks for asking!"}
]

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful and witty AI assistant inspired by JARVIS from the Iron"),
    MessagesPlaceholder(variable_name="{messages}"),
    ])






def chat_with_nikki():
    print("Chat with NIKKI (type 'quit' to end the conversation):")
    nikki = Ollama(model="llama2")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break
        response = nikki.invoke(user_input)
        print("NIKKI:", response)


# def chat_with_nikki():
#     llm = Ollama(model="llama2")
#     print(llm.invoke("Tell me a joke"))
    

# Example usage
if __name__ == "__main__":
    chat_with_nikki()


