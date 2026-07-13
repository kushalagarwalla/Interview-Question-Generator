import os
import json
import traceback
import pypdf

def read_file(file):
    if file.name.endswith(".pdf"):
        try:
            pdf_reader=pypdf.PdfReader(file)
            text=""
            for page in pdf_reader.pages:
                # extract_text() returns None on image-only pages
                text+=page.extract_text() or ""
            return text

        except Exception as e:
            raise Exception("Error reading the PDF file") from e

    elif file.name.endswith(".txt"):
        return file.read().decode("utf-8")

    else:
        raise Exception("Unsupported file format. Only PDF and TEXT files are supported")


def get_table_data(questions):
    try:
        question_table_data=[]

        #iterate over the questions dictionary and extract required information
        for key,value in questions.items():
            question_table_data.append({
                "Question":value["question"],
                "Answer":value["answer"],
                "Difficulty":value["difficulty"],
                "Category":value["category"],
                "Primary Skill":value["primary_skill"]
            })

        return question_table_data

    except Exception as e:
        traceback.print_exception(e)
        return None