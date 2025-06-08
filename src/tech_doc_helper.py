from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = "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"

# Load documentation
loader = TextLoader("docs/Final_Report.ipynb")
documents = loader.load()

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = splitter.split_documents(documents)

# Embed and store
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(texts, embeddings)
retriever = vectorstore.as_retriever()

# Set up LLM and chain
llm = OpenAI(model="gpt-4")
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

# Query the system
query = "What is the report about?"
answer = qa_chain.run(query)
print(answer)
