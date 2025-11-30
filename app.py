import os
import json
from typing import TypedDict, Annotated, List, Optional

from dotenv import load_dotenv

# LangChain Imports
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import (
    SystemMessage, HumanMessage, BaseMessage, AIMessage
)
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages

# Pydantic v2+
from pydantic import BaseModel, Field

# Local Import
from rag_pipeline import TransitRetriever


load_dotenv()
# --- LangSmith debug print (optional, but good for viva) ---
if os.getenv("LANGCHAIN_TRACING_V2") == "true":
    print(f"🛰 LangSmith tracing is ENABLED for project: {os.getenv('LANGCHAIN_PROJECT')}")
else:
    print("⚠ LangSmith tracing is NOT enabled. Set LANGCHAIN_TRACING_V2=true in .env")

# ------------------- Initialize RAG retriever -------------------
try:
    transit_db = TransitRetriever()
except Exception as e:
    print(f"Error initializing TransitRetriever: {e}")
    exit()


# ------------------- STRUCTURED BOOKING MODEL -------------------
class TicketBooking(BaseModel):
    train_name: str = Field(description="Name of the train")
    source: str = Field(description="Departure city")
    destination: str = Field(description="Arrival city")
    passenger_name: str = Field(description="Passenger name")
    class_type: str = Field(description="Travel class")


# ------------------- DEFINE TOOL -------------------
@tool
def search_railway_info(query: str):
    """
    Search train schedules + policies.
    """
    return transit_db.search(query)


tools = [search_railway_info]
tool_node = ToolNode(tools)

# LLM setup
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_with_tools = llm.bind_tools(tools)

# Structured output LLM for booking confirmation
structured_llm = llm.with_structured_output(TicketBooking)


# ------------------- GRAPH STATE -------------------
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


# ------------------- AGENT LOGIC NODE -------------------
def agent_logic(state: AgentState):
    messages = state["messages"]
    user_msg = messages[-1].content.lower()

    # ---- YES LOGIC: Show IRCTC link ----
    if user_msg in ["yes", "y", "book", "book it", "confirm", "ok book"]:
        return {
            "messages": [
                AIMessage(
                    content=(
                        "Great! You can book the ticket using the official IRCTC portal:\n"
                        "👉 **https://www.irctc.co.in/nget/train-search**\n\n"
                        "If you'd like, I can help you find more trains or compare prices!"
                    )
                )
            ]
        }

    # ---- NO LOGIC ----
    if user_msg in ["no", "nope", "nah"]:
        return {
            "messages": [
                AIMessage(
                    content="No problem! Tell me what else you want to search or ask."
                )
            ]
        }

    # ---- DEFAULT LOGIC (LLM decides, uses tools etc.) ----
    system = SystemMessage(
        content=(
            "You are TransitSense.\n"
            "1. Use 'search_railway_info' tool when user asks about schedules or policies.\n"
            "2. If user says 'book this' or 'confirm ticket', ask for passenger name if missing.\n"
            "3. When ready, output EXACT TEXT: READY_TO_BOOK.\n"
        )
    )

    response = llm_with_tools.invoke([system] + messages)
    return {"messages": [response]}


# ------------------- BOOKING PARSER NODE -------------------
def booking_parser(state: AgentState):
    messages = state["messages"]

    print("📝 Generating Final Ticket JSON...")

    result = structured_llm.invoke(messages)
    return {"messages": [AIMessage(content=str(result))]}


# ------------------- BUILD THE GRAPH -------------------
workflow = StateGraph(AgentState)

workflow.add_node("agent", agent_logic)
workflow.add_node("tools", tool_node)
workflow.add_node("booking", booking_parser)

workflow.set_entry_point("agent")

# Routing Logic
workflow.add_conditional_edges(
    "agent",
    lambda state: (
        "tools"
        if isinstance(state["messages"][-1], AIMessage)
        and state["messages"][-1].tool_calls
        else (
            "booking"
            if "READY_TO_BOOK" in state["messages"][-1].content
            else END
        )
    ),
    {
        "tools": "tools",
        "booking": "booking",
        END: END,
    },
)

# After tools → back to agent
workflow.add_edge("tools", "agent")

# Booking ends flow
workflow.add_edge("booking", END)

app = workflow.compile()


# ------------------- MAIN CHAT LOOP -------------------
if __name__ == "__main__":
    print("🚆 TransitSense AI Running...\n")

    state = {"messages": []}

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            break

        result = app.invoke({"messages": [HumanMessage(content=user_input)]})
        ai_msg = result["messages"][-1]
        print("\nAI:", ai_msg.content, "\n")
