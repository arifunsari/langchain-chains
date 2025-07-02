from langchain_openai import ChatOpenAI  # Load the OpenAI language model (LLM)
from langchain_anthropic import ChatAnthropic  # (Not used in this code) – used to load Anthropic models like Claude
from dotenv import load_dotenv  # Used to load environment variables from a .env file
from langchain_core.prompts import PromptTemplate  # For creating prompts with dynamic inputs
from langchain_core.output_parsers import StrOutputParser  # To parse plain string outputs from LLM
from langchain.schema.runnable import RunnableParallel, RunnableBranch, RunnableLambda  # For composing multiple chains in parallel/conditional branches
from langchain_core.output_parsers import PydanticOutputParser  # Parser that validates and structures output using Pydantic models
from pydantic import BaseModel, Field  # Used to define structured data models with validation
from typing import Literal  # Used to restrict a field to specific values, like "positive" or "negative"

load_dotenv()  # Load environment variables (e.g., OpenAI API key) from .env file

model = ChatOpenAI()  # Initialize the OpenAI chat model (e.g., gpt-3.5/gpt-4)

parser = StrOutputParser()  # Parser that returns plain text output from the model

# Define a Pydantic model to extract sentiment from feedback
class Feedback(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(description='Give the sentiment of the feedback')  # Restrict to only "positive" or "negative"

# Create a parser that will parse model output into the Feedback Pydantic model
parser2 = PydanticOutputParser(pydantic_object=Feedback)

# Prompt to classify the sentiment of a feedback (positive or negative), with format instructions
prompt1 = PromptTemplate(
    template='Classify the sentiment of the following feedback text into postive or negative \n {feedback} \n {format_instruction}',
    input_variables=['feedback'],  # Dynamic variable for feedback text
    partial_variables={'format_instruction': parser2.get_format_instructions()}  # Auto-generated instructions for expected Pydantic format
)

classifier_chain = prompt1 | model | parser2  # Chain: prompt -> model -> parse output into Feedback class

# Prompt for generating a response if the feedback is positive
prompt2 = PromptTemplate(
    template='Write an appropriate response to this positive feedback \n {feedback}',
    input_variables=['feedback']
)

# Prompt for generating a response if the feedback is negative
prompt3 = PromptTemplate(
    template='Write an appropriate response to this negative feedback \n {feedback}',
    input_variables=['feedback']
)

# Conditional branch: if sentiment is positive -> prompt2 | model | parser,
#                     if negative -> prompt3 | model | parser,
#                     else return fallback message
branch_chain = RunnableBranch(
    (lambda x: x.sentiment == 'positive', prompt2 | model | parser),  # Handle positive sentiment
    (lambda x: x.sentiment == 'negative', prompt3 | model | parser),  # Handle negative sentiment
    RunnableLambda(lambda x: "could not find sentiment")  # Fallback if sentiment is not classified
)

# Final chain: first classify sentiment, then use appropriate response generator based on sentiment
chain = classifier_chain | branch_chain

# Run the chain with sample feedback input
print(chain.invoke({'feedback': 'This is a beautiful phone'}))  # Input feedback to be classified and responded to

# Visualize the full chain as an ASCII graph (classifier_chain → branch_chain)
chain.get_graph().print_ascii()
