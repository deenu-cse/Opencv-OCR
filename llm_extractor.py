import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.1-8b-instant",
    temperature=0
)

def extract_fields(text: str, fields: list):
    prompt = PromptTemplate(
        input_variables=["text", "fields"],
        template="""
You are a document data extractor.

Extract the following fields from the given OCR text:
Fields: {fields}

Rules:
- If a field is not found, return N/A
- Return response strictly in JSON
- No explanation

OCR Text:
{text}
"""
    )

    chain = prompt | llm
    response = chain.invoke({
        "text": text,
        "fields": ", ".join(fields)
    })

    return response.content
