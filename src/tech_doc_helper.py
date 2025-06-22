from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import NotebookLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()


# Set OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")
if api_key is None:
    raise ValueError(
        "OPENAI_API_KEY environment variable not set."
        "Please set it in your .env file."
    )
os.environ["OPENAI_API_KEY"] = api_key

# --- Construct a portable path to the documentation file ---
# Get the directory of the current script (src/)
script_dir = Path(__file__).resolve().parent
# Get the project root directory (the parent of src/)
project_root = script_dir.parent
# Build the path to the notebook file
notebook_path = project_root / "docs" / "Final_Report.ipynb"

# Load documentation using the new, portable path
loader = NotebookLoader(str(notebook_path))
documents = loader.load()

# Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=100, separators=["\n\n", "\n", " ", ""]
)
texts = splitter.split_documents(documents)

# Embed and store
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(texts, embeddings)
retriever = vectorstore.as_retriever()

# Set up LLM and chain
llm = ChatOpenAI(model="gpt-4", temperature=0)
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

# Query the system
query = "Generate a report of models used in the document"
answer = qa_chain.invoke(query)
print(answer)
