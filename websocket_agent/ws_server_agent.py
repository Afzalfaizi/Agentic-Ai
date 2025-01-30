# Practice of websocket for realtime apps (chating apps)

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph, END
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_text_splitters import CharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain import hub
from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState
import os
from dotenv import load_dotenv
import asyncio

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Replace with your frontend's URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# LLM Setup
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))
loader = TextLoader("data.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=50)
texts = text_splitter.split_documents(documents)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
db = FAISS.from_documents(texts, embeddings)
retriever = db.as_retriever()

info_retriever = create_retriever_tool(
    retriever,
    "hotel_information_sender",
    "Searches hotel-related information from a provided vector store.",
)
tools = [info_retriever]
llm_with_tools = llm.bind_tools(tools)

sys_msg = (
    "You are Alexandra Hotel's virtual assistant. Your primary role is to provide accurate hotel-related information. "
    "Use specialized tools to retrieve detailed hotel data including room availability, amenities, and pricing."
)

# Graph setup
def assistant(state: MessagesState):
    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"][-10:])]}

builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")
memory = MemorySaver()
agent = builder.compile(checkpointer=memory)

@app.get("/")
async def root():
    return {"message": "FastAPI backend for hotel assistant is running!"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connection established")
    try:
        while True:
            data = await websocket.receive_text()
            print(f"Received from client: {data}")
            messages = [HumanMessage(content=data)]
            node_to_stream = 'assistant'
            config = {"configurable": {"thread_id": "1"}}
            await websocket.send_text("Processing your query...")
            try:
                async for event in agent.astream_events({"messages": messages}, config, version="v2"):
                    if (
                        event["event"] == "on_chat_model_stream"
                        and event["metadata"].get("langgraph_node", "") == node_to_stream
                    ):
                        chunk = event["data"]["chunk"].content
                        await websocket.send_text(chunk)
            except Exception as e:
                await websocket.send_text(f"Error: {e}")
    except WebSocketDisconnect:
        print("WebSocket disconnected")
