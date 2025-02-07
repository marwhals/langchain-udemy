import argparse

from dotenv import load_dotenv
from langchain.chains import LLMChain, SequentialChain
from langchain.llms import OpenAI  # Full choice of llms can be found here..............
from langchain.prompts import PromptTemplate

load_dotenv()

# parse some command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--task", default="return a list of numbers")
parser.add_argument("--language", default="python")
args = parser.parse_args()

# Replace when appropriate or use a stub
# Consider switching out for some other LLM or building my own...or using the model locally
# TODO ensure computation is offloded to GPU
llm = OpenAI()

code_prompt = PromptTemplate(
    template="Write a very short {language} function that will {task}",
    input_variables=["language", "task"]
)

test_prompt = PromptTemplate(
    input_variables=["language", "code"],
    template="Write a test for the following {language} code:\n{code}"
)

code_chain = LLMChain(
    llm=llm,
    prompt=code_prompt,
    output_key="code"
)
# TODO refactor this out of here
test_chain = LLMChain(
    llm=llm,
    prompt=test_prompt,
    output_key="test"
)

chain = SequentialChain(
    chains=[code_chain, test_chain],
    input_variables=["task", "language"],
    output_variables=["test", "code"]
)

# Test the model
prompt = "Tell us a joke"
response = llm.invoke(prompt)
print(response)
