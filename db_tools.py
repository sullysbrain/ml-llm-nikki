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


search = Tool(
    name="Websearch",
    func=search_function,
    description="Searches the internet and returns relevant results on any topic."
)

calculator = Tool(
    name="Calculator",
    func=calculator_function,
    description="Performs mathematical calculations including arithmetic and more."
)

# @tool
# def some_tool(s: str):
#     """prepends a string with the word 'Processed:'"""
#     processed_data = "Processed: " + s
#     return processed_data

# Tool collection
tools = [search, calculator]
