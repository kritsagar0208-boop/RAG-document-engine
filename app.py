import requests
import streamlit as st

st.set_page_config(
    page_title="RAG Document Q&A Engine", page_icon="📄", layout="centered"
)

st.title("📄 AI Document Q&A Engine")
st.caption(
    "Powered by FastAPI, ChromaDB, OpenAI `text-embedding-3-small`, and `gpt-4o-mini`"
)

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display prior chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask a question about your uploaded document..."):
    # Add user message to display chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call FastAPI backend
    with st.chat_message("assistant"):
        with st.spinner("Searching document & generating answer..."):
            try:
                response = requests.get(
                    "http://127.0.0.1:8000/ask",
                    params={"query": prompt},
                    timeout=30,
                )
                if response.status_code == 200:
                    answer = response.json().get("answer", "No answer found.")
                else:
                    answer = f"Error: Received status code {response.status_code} from backend."
            except Exception as e:
                answer = f"Could not connect to FastAPI server. Make sure `uvicorn` is running! Details: {e}"

            st.markdown(answer)

    # Add assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": answer})