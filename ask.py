
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
def load_qa_chain(debug=False):
   embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
   vectorstore = Chroma(
       persist_directory="chroma_db",
       embedding_function=embedding
   )
   prompt_template = """
   You are an assistant for internal company queries.
   Use the following context to answer the question.
   If the answer is not in the context, just say you don't know.

   Information:
   {context}

   Question: {question}
   Answer in a consise and clear manner
   """
   prompt = PromptTemplate(
       template=prompt_template,
       input_variables=["context", "question"]
   )
   llm = Ollama(model="tinyllama", temperature=0.2, num_predict=256)  # Still okay — this is for *generation*, not embeddings
   return RetrievalQA.from_chain_type(
       llm=llm,
       retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
       chain_type_kwargs={"prompt": prompt},
       return_source_documents=debug
   )
