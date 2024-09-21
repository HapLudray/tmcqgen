import os
import json
import traceback
import pandas as pd
from dotenv import load_dotenv
from src.mcqgenerator.utils import read_file, get_table_data
from src.mcqgenerator.logger import logging

#import necessary packages from langchain
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain

load_dotenv()

KEY=os.getenv("OPENAI_API_KEY")

llm=ChatOpenAI(openai_api_key=KEY,model_name="gpt-4o",temperature=0.5)

TEMPLATE="""
Text:{text}
You are an expert MCQ maker. Given the above text, it is your job to \
create a quiz for {number} multiple-choice questions for {subject} students in {tone} tone.
Make sure the questions are not repeated and check all the questions to be comfirming the text as well.
Make sure to generate the response in the format like below and it should not contain any extra word like 'json'.\

{response_json}

"""

quiz_generation_prompt = PromptTemplate(
    input_variables=["text","number","subject","tone","response_json"],
    template = TEMPLATE
)

quiz_chain=LLMChain(llm=llm, prompt=quiz_generation_prompt, output_key="quiz", verbose=True)

TEMPLATE2="""
You are an expert English grammarian and writer. Given a Multiple-Choice Quiz for {subject} students.\
You need to evaluate the complexity of the question and give a complete analysis of the quiz. Only use at max 50 words for complexity.
If the quiz is not at per with the cognitive and analytical abilities of the students,\
update the quiz questions which needs to be changed and change the tone such that it perfectly fits the students.
Quiz_MCQs:
{quiz}

Check from an expert English Writer of the above quiz:
"""

quiz_evaluation_prompt = PromptTemplate(
    input_variables=["subject","quiz"],
    template = TEMPLATE2
)

review_chain = LLMChain(llm=llm, prompt=quiz_evaluation_prompt, output_key="review", verbose=True)

generate_evaluate_chain=SequentialChain(chains=[quiz_chain,review_chain], 
                                        input_variables=["text","number","subject","tone","response_json"],
                                        output_variables=["quiz","review"], verbose=True)

