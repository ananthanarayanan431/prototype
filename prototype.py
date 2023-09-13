
import os
import openai
print(openai.__version__)

import langchain
import streamlit as st
from constant import openai_key

from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain
from langchain.chains import SequentialChain
from langchain.memory import ConversationBufferMemory

os.environ['OPENAI_API_KEY']= openai_key

print(langchain.__version__)
print(st.__version__)
# print(openai_key)

#streamlit framework

st.title("Chatbot to respond to text queries pertaining the Mining Industries")
input_text = st.text_input("Enter the Topic you want to Search : ")

#Prompt Template

first_prompt = PromptTemplate(
    input_variables = ['name'],
    template = "Write a information about this act {name} "
)

person_memory = ConversationBufferMemory(input_key='name',memory_key="name_hist")
dob_memory = ConversationBufferMemory(input_key='person',memory_key="person_hist")
desc_memory = ConversationBufferMemory(input_key="DOB",memory_key="desc_hist")

#OPENAI API
llm = OpenAI(temperature=0.8)
chain = LLMChain(llm=llm,prompt=first_prompt,verbose=True,output_key="person",memory=person_memory)

second_prompt=PromptTemplate(
    input_variables = ['person'],
    template ="When this act was {person} Enacted and Enacted by"
)
chain2 = LLMChain(llm=llm,prompt=second_prompt,verbose=True,output_key="DOB",memory=dob_memory)


third_prompt = PromptTemplate(
    input_variables = ['DOB'],
    # template = "Mention top 5 event on that Day {DOB} in the World"
    template = "Mention top 5 interesting things about this act {DOB}"
)

chain3 = LLMChain(llm=llm,prompt=third_prompt,verbose=True,output_key="Description",memory=desc_memory)


parent_chain = SequentialChain(chains=[chain,chain2,chain3],input_variables=['name'],
                               output_variables=['person','DOB','Description'],
                               verbose=True
)

if input_text:
    # st.write(llm(input_text))
    st.write(parent_chain({'name':input_text}))

    with st.expander('Person Name'):
        st.info(person_memory.buffer)

    with st.expander('Major Event'):
        st.info(desc_memory.buffer)