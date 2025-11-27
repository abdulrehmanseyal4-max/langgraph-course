from dotenv import load_dotenv
from langgraph.graph import MessagesState, StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage

from nodes import tool_node, run_agent_reasoning

load_dotenv()

AGENT_REASON="agent_reason"
ACT="act"
LAST = -1

def should_continue(state: MessagesState) -> str:
    last_message = state["messages"][LAST]

    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return ACT
    return END

flow = StateGraph(MessagesState)

flow.add_node(AGENT_REASON, run_agent_reasoning)
flow.set_entry_point(AGENT_REASON)
flow.add_node(ACT, tool_node)

flow.add_conditional_edges(AGENT_REASON, should_continue, {
    END:END,
    ACT:ACT
})

flow.add_edge(ACT, AGENT_REASON)

app = flow.compile()
app.get_graph().draw_mermaid_png(output_file_path="flow.png")

def main():
    print("Hello from langgraph-course!")
    res = app.invoke({"messages": [HumanMessage(content="What is the temperature in Tokyo? List it and then triple it")]})
    print(res["messages"][LAST].content)

if __name__ == "__main__":
    main()
