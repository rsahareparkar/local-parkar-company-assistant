
import streamlit as st
from ask import load_qa_chain
st.set_page_config(page_title="ParkAI - Parkar's AI Assistant", page_icon="💼", layout="wide")
st.title("💼 ParkAI Assistant")
st.subheader("Ask me anything about Parkar's policy.")
query = st.text_input("Ask a question 👇")
if query:
   qa_chain = load_qa_chain(debug=True)
   response = qa_chain({"query": query})
   st.write("Answer:", response["result"])
   print(response)
   #st.write("🧠 Answer:")
   #st.write(response)
