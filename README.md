ParkAI Assistant:

Here’s a complete and clean README.md and an approach summary tailored specifically to your project using:
• Streamlit for UI
• Ollama with TinyLlama for local LLM
• LangChain for retrieval
• ChromaDB for vector search
• MiniLM embeddings (from HuggingFace)
# 💼 ParkAI – Local HR Assistant using LangChain + Ollama

ParkAI is an AI-powered internal assistant that answers employee questions by reading company documents (like PDFs). It's fully **offline**, runs on your **local machine**, and requires **no API keys**.

Example questions:
- “What is the notice period?”
- “Who should I contact for laptop issues?”

---

## 🧠 Features

- 🔍 Context-aware answers based on your internal PDF documents
- 💾 Uses `ChromaDB` for fast vector search
- 🧠 Runs `TinyLlama` LLM via [Ollama](https://ollama.com/)
- 💬 Simple web-based UI with Streamlit
- ✅ 100% offline, private, and open source

## 📁 Project Structure
Ollama
Get up and running with large language models.
.
├── app.py                  # Streamlit app frontend
├── ask.py                  # Handles querying the model
├── load_documents.py       # Loads and embeds PDFs into ChromaDB
├── data/                   # Folder containing internal company PDFs
├── chroma_db/              # Vector DB (auto-created)
└── README.md               # Project documentation
 
---

## 🚀 How to Run

### 1. 📦 Install Python Libraries

Create a virtual environment and install requirements:

```bash
pip install -r requirements.txt

pip install streamlit langchain chromadb sentence-transformers llama-cpp-python pypdf tqdm
 
2. 📄 Add Your Company Documents

Place all your internal PDFs inside the data/ folder
(e.g., data/company_policy.pdf)

3. 🧠 Ingest Documents

This step reads PDFs, splits them, and saves embeddings in chroma_db/.
 python load_documents.py 
You should see messages like:

✅ Loaded 10 pages from PDFs.
✅ Split into 80 chunks.
🔄 Embedding...
✅ Vectorstore saved to disk.
 
4. 🤖 Ensure Ollama is Running

Install Ollama and start TinyLlama:

Bash: ollama run tinyllama
 You can replace tinyllama with mistral, llama3, etc. if desired — just update it in ask.py. 
5. 💬 Start the Web UI

Bash: streamlit run app.py

Visit: http://localhost:8501

Type any question based on the document — Tell me about parkar’s certification Policy
 
🧠 Approach Summary

🔹 Document Loading
• All PDFs in the data/ folder are loaded using PyPDFLoader
• Each page is split into manageable chunks using RecursiveCharacterTextSplitter

🔹 Embedding
• Chunks are converted into vector embeddings using all-MiniLM-L6-v2 from HuggingFace
• The vectors and metadata are saved into ChromaDB (local persistent vector store)

🔹 Local LLM + Retrieval
• When a question is asked, a RetrievalQA chain is created using LangChain
• The relevant chunks are retrieved from ChromaDB
• The answer is generated by TinyLlama running locally via Ollama

🔹 Web UI
• Streamlit provides a user-friendly interface to input questions and view answers 
Example Output

Q: What is the notice period?
A:The notice period is the number of days' notice required for an employee to terminate their employment services with a company, unless the termination is on account of one of the grounds described under Involuuntary Separaion. The notice period can be served in either 15 or 60 days, depending on the company's policy. In case the company grants the employee a waiver or reduction on the notice period, there will be a recovery for the balance days' payment to the employee in lieu of the notice period. The company has the right to waive off the notice period in case of resignation/termination. Accrued earned leaves will be adjusted against the notice period.
 
🔐 No API Keys Required

This project is fully open-source and private:
• All data is processed locally
• No calls to OpenAI or external APIs
• Ideal for internal company usage (HR, IT, onboarding, etc.)

⸻

🙌 Credits
• LangChain
• Ollama
• HuggingFace Transformers
• Streamlit
 
==================================EoD===================================
