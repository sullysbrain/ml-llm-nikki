"""
File: db_tools.py
Author: Scott Sullivan
Created: 2024-05-02
Description:
    This module is a collection of @tools for building the conversational AI chain.

Functions:

"""
from langchain.tools import Tool
from langchain.agents import load_tools
from langchain_core.tools import tool

@tool
def search_function(query: str):
    """Searches the internet and returns relevant results on any topic."""
    import requests
    from bs4 import BeautifulSoup

    url = f"https://www.google.com/search?q={query}"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    results = soup.find_all("div", class_="BNeawe iBp4i AP7Wnd")
    return results[0].get_text()

@tool
def add_function(first_int: int, second_int: int) -> int:
    """Performs mathematical calculations including arithmetic and more."""
    return first_int + second_int

search = Tool(
    name="Websearch",
    func=search_function,
    description="Searches the internet and returns relevant results on any topic."
)

add = Tool(
    name="Calculator",
    func=add_function,
    description="Performs mathematical calculations including arithmetic and more."
)


# @tool
# def some_tool(s: str):
#     """prepends a string with the word 'Processed:'"""
#     processed_data = "Processed: " + s
#     return processed_data

# Tool collection
tools = [search, add]

