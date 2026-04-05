import streamlit as st
from backend.rag_engine import get_answer_from_gemini as get_answer_from_llm 


st.set_page_config(page_title="GitLab RAG Assistant", page_icon="🦊", layout="centered")
st.title("🦊 GitLab Handbook Assistant")
st.markdown("Ask me anything about GitLab's culture, values, or direction based on their public handbook.")


if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if user_input := st.chat_input("E.g., What are GitLab's core values?"):
    

    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        with st.spinner("Searching the database..."):
            

            history_to_pass = st.session_state.messages[:-1]
            ai_response = get_answer_from_llm(user_input, history_to_pass) 
            
        st.markdown(ai_response)
        

    st.session_state.messages.append({"role": "assistant", "content": ai_response})