from setuptools import find_packages, setup

setup(
    name="InterviewQuestionGenerator",
    version="0.0.1",
    author="Kushal Agarwalla",
    author_email="kushal.agarwal72@gmail.com",
    install_requires=[
        "openai",
        "langchain",
        "langchain-openai",
        "pypdf",
        "streamlit",
        "python-dotenv",
    ],
    packages=find_packages()

)