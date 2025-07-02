from langchain_openai import ChatOpenAI  #importing the LLM
from dotenv import load_dotenv # to load the .env file like fetch the API
from langchain_core.prompts import PromptTemplate # from langchain_core.prompts import PromptTemplate to create the prompt.
from langchain_core.output_parsers import StrOutputParser # to initialize the data type of the output in the simple string.

load_dotenv() # now load the .env file

prompt1 = PromptTemplate(  # create the template
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate( # create the another template from the prompt1 output
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)

model = ChatOpenAI()  # initialize any LLM model from ChatOpenAI

parser = StrOutputParser()  # now initilaze the parser

chain = prompt1 | model | parser | prompt2 | model | parser # create the fllow of executions

result = chain.invoke({'topic': 'Unemployment in India'}) # invoke means call it.

print(result)

chain.get_graph().print_ascii()  #to see the chians in th visual format

# Is code me hmne phle LLM se ek detailed answer nikala hai prompt1 se, then usi detailed answer se 5 pointer summary nikala hai prompt 2 se,
# usse chain ke madad se integrate kiya hai.