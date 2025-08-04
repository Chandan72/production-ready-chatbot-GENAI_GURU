from dotenv import load_dotenv
import os
load_dotenv()
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import trim_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph



#  initialize Google API key and model
google_api_key=os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY is not set in the environment variables.") 

model= init_chat_model("gemini-2.5-pro",model_provider="google_genai", api_key=google_api_key)
# Define the system prompt and message template
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

system_prompt = (
    "You are GenAI-Guru, an expert assistant focused EXCLUSIVELY on Generative AI. "
    "Answer questions ONLY related to LLMs, prompt engineering, text/image/audio models, and their research or application. "
    "If a question is NOT about Generative AI, state: 'Sorry, I only answer Generative AI questions.' "
    "Provide clear, modern explanations; cite research, libraries, or APIs where relevant. "
    "Include concise code where appropriate, using Python, LangChain, or HuggingFace. "
    "Do NOT provide medical, legal, financial, or general tech advice. "
    "Do NOT answer jokes, trivia, or opinion-seeking out-of-domain prompts. "
    "Stay factual. If you don't know, say so honestly."
)
prompt_template= ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="messages"),
    ("human", "{input}")

])
# set up the trimming part
trimmer = trim_messages(max_tokens=1800, 
                        max_messages=10, 
                        strategy="last", 
                        token_counter=model, 
                        include_system=True, 
                        allow_partial=False)

# define the model calling function
def call_model(state: MessagesState):
    trimmed=trimmer.invoke(state["messages"])
    prompt=prompt_template.invoke({"messages": trimmed })
    response=model.invoke(prompt)
    return {"messages": state["messages"]+ [response]}

# stategeaph manages memory and connection to LLM

workflow = StateGraph(state_schema=MessagesState)
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)
memory= MemorySaver()
app=workflow.compile(checkpointer=memory)


def get_bot_response(user_message, thread_id="default-session"):
    config ={"configurable": {"thread_id": thread_id}}
    input_message = [HumanMessage(content=user_message)]
    output = app.invoke({"messages": input_message}, config=config)
    messages=output["messages"]
    return output["messages"][-1].content if messages else "No response generated."

def stream_bot_response(user_message, thread_id="default-session"):
    config = {"configurable": {"thread_id": thread_id}}
    from langchain_core.messages import HumanMessage
    input_messages = [HumanMessage(content=user_message)]
    # This is a generator yielding content as it’s produced
    for chunk, meta in app.stream(
        {"messages": input_messages}, config, stream_mode="messages"
    ):
        if hasattr(chunk, "content"):
            yield chunk.content



