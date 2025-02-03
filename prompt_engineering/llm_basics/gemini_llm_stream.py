from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import sys
load_dotenv()


llm = GoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
    )

question = input("Enter you prompt: make a exercise plan for 1 week according to my weight  ")

for chunk in llm.stream(question):
    print(chunk)