import streamlit as st
from rag import RagMultimodal


@st.cache_resource
def initialization_function():
    rag = RagMultimodal()
    return rag


rag = initialization_function()

st.title("Rag")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if query := st.chat_input("What is up?"):
    st.chat_message("user").markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})

    response = rag.chain.invoke(query)
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
