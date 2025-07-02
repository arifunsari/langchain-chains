from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel # multiple chains ko parallelly ek sath execute kar skte hai.

load_dotenv()

model1 = ChatOpenAI() # model 1 for Notes

model2 = ChatAnthropic(model_name='claude-3-7-sonnet-20250219') # model 2 for QUIZ

prompt1 = PromptTemplate(  # Template 1 for Notes
    template='Generate short and simple notes from the following text \n {text}',
    input_variables=['text']
)

prompt2 = PromptTemplate( # Template 2 for Quiz
    template='Generate 5 short question answers from the following text \n {text}',
    input_variables=['text']
)

prompt3 = PromptTemplate( # Template 3 two merze notes and quiz in single doc.
    template='Merge the provided notes and quiz into a single document \n notes -> {notes} and quiz -> {quiz}',
    input_variables=['notes', 'quiz'] # two input variables notes and quiz
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({  # dono chains ko ek naam deni hai like 
    'notes': prompt1 | model1 | parser, # chain 1 = notes
    'quiz': prompt2 | model2 | parser # chain 2 = quiz
})

# parallel chain ho gaya upeer me, now we will write the logic of mergin chain in  down.
merge_chain = prompt3 | model1 | parser # now merze two chains

chain = parallel_chain | merge_chain # chain 3 to merze the chain 1 and chain 2

text = """
Support vector machines (SVMs) are a set of supervised learning methods used for classification, regression and outliers detection.

The advantages of support vector machines are:

Effective in high dimensional spaces.

Still effective in cases where number of dimensions is greater than the number of samples.

Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.

Versatile: different Kernel functions can be specified for the decision function. Common kernels are provided, but it is also possible to specify custom kernels.

The disadvantages of support vector machines include:

If the number of features is much greater than the number of samples, avoid over-fitting in choosing Kernel functions and regularization term is crucial.

SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation (see Scores and probabilities, below).

The support vector machines in scikit-learn support both dense (numpy.ndarray and convertible to that by numpy.asarray) and sparse (any scipy.sparse) sample vectors as input. However, to use an SVM to make predictions for sparse data, it must have been fit on such data. For optimal performance, use C-ordered numpy.ndarray (dense) or scipy.sparse.csr_matrix (sparse) with dtype=float64.
"""

result = chain.invoke({'text':text}) # invoke the chains

print(result)

chain.get_graph().print_ascii()


# Pareele code me hame like ek doc mila usme Transformer pe detailed notes hai, jisme se hame ek hi samay pe usse NOTES aur QUIZ genrate krna hai,
# we do like mode1 - Notes, model2 - QUIZE then end me dono ko comine kar ke model3- se dono ka combination user ko show karege.
