import os

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_mistralai import ChatMistralAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt
from agent_tools.tools import run_python, search_docs

tools = [search_docs, run_python]

load_dotenv()

# Stop if MISTRAL_API_KEY is not set in the environment variables.
if not os.getenv("MISTRAL_API_KEY"):
    raise RuntimeError("MISTRAL_API_KEY manquante dans .env")

SYSTEM_PROMPT = (
    "Utilise l'outil search_docs UNIQUEMENT si tu as besoin d'informations de la documentation. "
    "Utilise run_python UNIQUEMENT pour exécuter du code Python. "
    "Si tu peux répondre directement, fais-le sans appeler d'outil."
)
# Init Model
model = ChatMistralAI(model="mistral-small-latest", temperature=0)

bound_model = model.bind_tools(tools)
class AgentState(MessagesState):
    validated: bool

# Noeud
def agent_node(state: AgentState) -> dict:
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    for attempt in range(5):
        try:
            response = bound_model.invoke(messages)
            return {"messages": [response]}
        except Exception as e:
            if "tool_use_failed" in str(e) and attempt < 5:
                print(f"[agent] Mistral tool_use_failed, retry {attempt + 1}/5...")
                continue
            raise

# Nœud d'interruption humaine
def validate_node(state: AgentState) -> dict:
    last_message = state["messages"][-1]
    
    tool_call = last_message.tool_calls[0]
    
    decision = interrupt({
        "question": "Voulez-vous autoriser l'execution du script python ?",
        "tool_call_id": tool_call["id"]
    })
    
    print("Décision reçue de l'humain :", decision)
    return {"validated": decision in ["Oui", "oui", "yes", "Yes", "YES", "True", "true", "TRUE", "1"]}

def tools_node(state: dict) -> dict:
    last_message = state["messages"][-1] # L'AIMessage avec l'appel d'outil
    tool_call = last_message.tool_calls[0]
    
    if state.get("validated", False):
        res = run_python.invoke(tool_call["args"])
    else:
        res = "L'utilisateur n'a pas approuvé l'exécution du script python."
        
    tool_msg = ToolMessage(content=res, tool_call_id=tool_call["id"], name=tool_call["name"])
    return {"messages": [tool_msg]}

def reject_node(state: AgentState) -> dict:
    res = "Annulé : l'exécution du script python a été refusée."
    last_message = state["messages"][-1]
    tool_call = last_message.tool_calls[0]
    tool_msg = ToolMessage(content=res, tool_call_id=tool_call["id"], name=tool_call["name"])
    return {"messages": [tool_msg]}

# Routeur conditionnel
def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        tool_name = last_message.tool_calls[0]["name"]
        if tool_name == "run_python":
            return "validate"
        return "tools"
    return END

def approval_route(state: AgentState) -> str:
    if (state.get("validated", False)):
        return "tools"
    return "reject"

search_tool_node = ToolNode([search_docs])

builder = StateGraph(AgentState)

builder.add_node("agent", agent_node)
builder.add_node("validate", validate_node)
builder.add_node("tools", tools_node)
builder.add_node("search", search_tool_node)
builder.add_node("reject", reject_node)

builder.add_edge(START, "agent")
builder.add_conditional_edges(
    "agent",
    should_continue,
    {
        "validate": "validate",
        "tools": "search",
        END: END
    }
)
builder.add_conditional_edges(
    "validate",
    approval_route,
    {"tools": "tools", "reject": "reject"}
)

builder.add_edge("tools", "agent")
builder.add_edge("search", "agent")
builder.add_edge("reject", "agent")

graph = builder.compile()

# Dev testing
# checkpointer = MemorySaver()
# graph = builder.compile(checkpointer=checkpointer)
# config = {"configurable": {"thread_id": "thread-01"}}

# print("=== Lancement initial ===")
# events = graph.stream(
#     {"messages": [HumanMessage(content="Explique moi l'interrupt en langgraph")]}, 
#     config, 
#     stream_mode="values"
# )
# for ev in events:
#     print(ev)

# # On vérifie que le graphe est bien en attente d'une interruption
# state_info = graph.get_state(config)
# print("Statut d'interruption :", state_info.tasks)
# print("Valeur de l'interruption :", state_info.tasks[0].interrupts if state_info.tasks else "Aucun")

# print("=== Lancement avec interrupt ===")
# config2 = {"configurable": {"thread_id": "thread-02"}}
# events = graph.stream(
#     {"messages": [HumanMessage(content="Fait et test un script python qui calcule 2 + 2")]}, 
#     config2, 
#     stream_mode="values"
# )
# for ev in events:
#     print(ev)

# # On vérifie que le graphe est bien en attente d'une interruption
# state_info = graph.get_state(config)
# print("Statut d'interruption :", state_info.tasks)
# print("Valeur de l'interruption :", state_info.tasks[0].interrupts if state_info.tasks else "Aucun")