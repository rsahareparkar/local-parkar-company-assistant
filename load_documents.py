
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document
from tqdm import tqdm
# 📂 Step 1: Load all PDFs from data folder
folder_path = "data"
documents = []
for filename in os.listdir(folder_path):
   if filename.endswith(".pdf"):
       file_path = os.path.join(folder_path, filename)
       loader = PyPDFLoader(file_path)
       pdf_pages = loader.load()
       documents.extend(pdf_pages)
print(f"✅ Loaded {len(documents)} pages from PDFs.")
# ✂️ Step 2: Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
   chunk_size=1000,
   chunk_overlap=200
)
chunks = text_splitter.split_documents(documents)
print(f"✅ Split into {len(chunks)} chunks.")
# 🧠 Step 3: Embed with sentence-transformers
print("🧠 Starting embedding with HuggingFace model...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
texts = [chunk.page_content for chunk in chunks]
metadatas = [chunk.metadata for chunk in chunks]
# 🔄 Optional: Progress bar
print("🔄 Embedding chunks with progress...")
documents = []
for text, meta in tqdm(zip(texts, metadatas), total=len(texts), desc="🔄 Embedding"):
   doc = Document(page_content=text, metadata=meta)
   documents.append(doc)
# 💾 Step 4: Save to Chroma
print("💾 Saving to Chroma...")
vectorstore = Chroma.from_documents(
   documents=documents,
   embedding=embeddings,
   persist_directory="chroma_db"
)
# 🧪 Step 5: Run test query
print("🧪 Running test query...")
results = vectorstore.similarity_search("test", k=1)
print(f"✅ Vectorstore test search returned {len(results)} result(s)")
vectorstore.persist()
print("✅ Vectorstore saved to disk.")
