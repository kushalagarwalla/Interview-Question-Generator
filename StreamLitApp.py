import os
import json
import pandas as pd
import traceback
from dotenv import load_dotenv
import streamlit as st
from src.InterviewQuestionGenerator.InterviewQuestionGenerator import evaluation_chain
from src.InterviewQuestionGenerator.logger import logging
from src.InterviewQuestionGenerator.utils import read_file, get_table_data
from langchain_core.callbacks import get_usage_metadata_callback

#loading JSON file
with open('Response.json','r') as file:
    RESPONSE_JSON=json.load(file)

#creating title for the app
st.title('Interview Question Generator')

#creating a form using st.form
with st.form("user_inputs"):
    #File Upload
    uploaded_file=st.file_uploader("Upload a PDF or TEXT file")

    #Input Fields
    question_count=st.number_input("Number of Questions", min_value=3, max_value=30)
    domain=st.text_input("Domain",max_chars=50)
    seniority=st.text_input("Seniority Level",max_chars=20,placeholder="Entry Level")

    #Add Button
    button=st.form_submit_button("Generate Interview Questions")

    #Check if button is clicked and all fields have inputs
    if button and uploaded_file is not None and question_count and domain and seniority:
        with st.spinner("loading..."):
            try:
                text=read_file(uploaded_file)
                logging.info(f"Read {len(text)} characters from {uploaded_file.name}")

                #count tokens and cost of API Calls
                with get_usage_metadata_callback() as cb:
                    response=evaluation_chain.invoke(
                        {
                            "text":text,
                            "question_count":question_count,
                            "domain":domain,
                            "seniority":seniority,
                            "response_json":json.dumps(RESPONSE_JSON)
                        }
                    )
            except Exception as e:
                traceback.print_exception(e)
                logging.exception("Chain invocation failed")
                st.error(f"Something went wrong: {e}")

            else:
                logging.info(f"Token usage: {cb.usage_metadata}")
                print(f"Total Tokens: {cb.usage_metadata}")

                if isinstance(response,dict):
                    #Extract questions from the response
                    questions=response.get("questions",None)
                    if questions is not None:
                        table_data=get_table_data(questions)
                        if table_data is not None:
                            df=pd.DataFrame(table_data)
                            df.index=df.index+1
                            st.table(df)

                            #Display the review in a text box
                            st.text_area(label="Review",value=response["review"])

                        else:
                            st.error("Error in table data")

                else:
                    st.write(response)