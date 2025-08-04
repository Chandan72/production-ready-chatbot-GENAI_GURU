import streamlit as st
from chatbot_core import stream_bot_response  # streaming function for smooth chat

st.set_page_config(page_title="GenAI Guru Chatbot", page_icon="🤖")
st.title("🤖 GenAI Guru: Exclusive Generative AI Assistant")
st.write("Ask only about LLMs, prompt engineering, and generative models.")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display previous chat history
for message in st.session_state["messages"]:
    with st.chat_message("user"):
        st.markdown(message["user"])
    with st.chat_message("assistant"):
        st.markdown(message["bot"])

# Input for the user's next question
if user_query := st.chat_input("Type your Generative AI question..."):
    st.session_state["messages"].append({"user": user_query, "bot": ""})
    with st.chat_message("assistant"):
        result_slot = st.empty()
        answer = ""
        # Stream the answer chunk by chunk as Gemini responds
        for chunk in stream_bot_response(user_query):
            answer += chunk
            result_slot.markdown(answer)
        st.session_state["messages"][-1]["bot"] = answer

st.sidebar.write("This chatbot answers only Generative AI questions. [Reset chat](#)")
if st.sidebar.button("Start new chat"):
    st.session_state["messages"] = []
    st.experimental_rerun()
