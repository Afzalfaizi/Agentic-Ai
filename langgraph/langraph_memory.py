from typing_extensions import TypedDict
from typing import Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI
from fastapi import FastAPI
from uuid import uuid4  # To generate unique thread IDs
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))

class MessagesState(TypedDict):
    messages: Annotated[list, add_messages]

def assistant(state: MessagesState):
    return {"messages": [llm.invoke(state["messages"])]}

# Build graph
builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_edge(START, "assistant")
builder.add_edge("assistant", END)
checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

app = FastAPI()

@app.get("/chat/{query}")
def get_content(query: str, thread_id: str = None):
    print(query)
    try:
        # If thread_id is not passed, generate a new unique thread_id using uuid4
        thread_id = thread_id or str(uuid4())
        
        # Update the config with the dynamic thread_id
        config = {"configurable": {"thread_id": thread_id}}
        
        result = graph.invoke({"messages": [("user", query)]}, config)
        return result
    except Exception as e:
        return {"output": str(e)}

# poetry run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
