from dotenv import load_dotenv
from sqlmodel import create_engine, SQLModel, Field, Session, select
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from typing import Optional
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph, END
from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.graph import MessagesState

# Load environment variables
load_dotenv()

# Database setup

class Student(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    email: str
    phone: str
    class_name: str


connection_string: str = str(os.getenv("DATABASE_URI")).replace("postgresql", "postgresql+psycopg")

engine = create_engine(connection_string, connect_args={"sslmode": "require"}, pool_recycle=3600, pool_size=10, echo=True)

def create_tables():
    SQLModel.metadata.create_all(engine)

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Creating Tables")
    create_tables()
    print("Tables Created")
    try:
        yield
    finally:
        print("Lifespan context ended")

# FastAPI app initialization
app = FastAPI(lifespan=lifespan)

@app.get('/')
def index():
    return {"message": "Welcome to My AI College Management System"}

# CRUD Operations
# CRUD Operations with proper docstrings
def add_student(name: str, email: str, phone: str, class_name: str) -> Student:
    """
    Add a new student to the database.
    
    Args:
        name (str): The name of the student.
        email (str): The email of the student.
        phone (str): The phone number of the student.
        class_name (str): The class name of the student.
        

    Returns:
        Student: The created student object.
    """
    student = Student(name=name, email=email, phone=phone, class_name=class_name)
    with Session(engine) as session:
        session.add(student)
        session.commit()
        session.refresh(student)
    return student

def read_todos(status: str = None) -> list[Todo]:
    """
    Retrieve todos from the database.

    Args:
        status (str, optional): Status to filter todos. Defaults to None.

    Returns:
        list[Todo]: List of todos matching the status filter.
    """
    with Session(engine) as session:
        statement = select(Todo)
        if status:
            statement = statement.where(Todo.status == status)
        todos = session.exec(statement).all()
    return todos

def update_todo(todo_id: int, title: str = None, description: str = None, status: str = None) -> Todo:
    """
    Update a todo in the database.

    Args:
        todo_id (int): ID of the todo to update.
        title (str, optional): New title of the todo. Defaults to None.
        description (str, optional): New description of the todo. Defaults to None.
        status (str, optional): New status of the todo. Defaults to None.

    Returns:
        Todo: The updated todo object, or None if not found.
    """
    with Session(engine) as session:
        todo = session.get(Todo, todo_id)
        if not todo:
            return None
        if title:
            todo.title = title
        if description:
            todo.description = description
        if status:
            todo.status = status
        session.add(todo)
        session.commit()
        session.refresh(todo)
    return todo

def delete_todo(todo_id: int) -> bool:
    """
    Delete a todo from the database.

    Args:
        todo_id (int): ID of the todo to delete.

    Returns:
        bool: True if deleted successfully, False if not found.
    """
    with Session(engine) as session:
        todo = session.get(Todo, todo_id)
        if not todo:
            return False
        session.delete(todo)
        session.commit()
    return True


# LLM setup
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))
tools = [add_student]
llm_with_tools = llm.bind_tools(tools)

# System message
sys_msg = """
You are a college Management Assistant with access to tools for managing adding students in database. You can perform the following actions:

- **add student Todo**: Add a new students by providing a name, an email address, phone number, and class name.

### Guidelines:
- Always ask for the required details to perform an action and confirm completion with clear feedback.
- Keep your responses short, focused, and task-oriented. Avoid unnecessary or irrelevant information.
- Use the provided tools to efficiently perform actions. Do not attempt tasks that can be handled using external tools.
- Handle errors with empathy and politely inform the user about any issues.
- Stay within the scope of todo management. If asked about unrelated topics, kindly remind the user of your purpose and steer the conversation back to college management.

Maintain a professional, polite, and helpful tone throughout your interactions.
"""

# Assistant definition
def assistant(state: MessagesState):
    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"][-10:])]}  # Include recent messages

# Graph nodes and edges
builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")

# Graph memory
memory = MemorySaver()

# Build the graph
agent = builder.compile(checkpointer=memory)

# Chat API Endpoint
@app.get("/chat/{query}")
def get_content(query: str):
    """
    Process chat queries and return responses
    Uses a fixed thread_id for demonstration purposes
    """
    try:
        config = {"configurable": {"thread_id": "2"}}
        result = agent.invoke({"messages": [("user", query)]}, config)
        return result
    except Exception as e:
        return {"output": str(e)}