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


