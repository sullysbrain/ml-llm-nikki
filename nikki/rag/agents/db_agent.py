# This is a tool that can be used in the langchain
from langchain.tools import BaseTool


# This is a custom logic that can be used in the langchain tool
def calculate_string(input_string):
    # Custom logic to process the input string
    # For demonstration, let's say it returns the length of the string
    return len(input_string)



class CalculateStringTool(BaseTool):
    def __init__(self):
        super().__init__()

    def __call__(self, input_string):
        tmp = calculate_string(input_string)
        return "The length of the input string is: " + str(tmp)

    @property
    def name(self):
        return "calculate_string"

    @property
    def description(self):
        return "Calculates the length of the given input string."
