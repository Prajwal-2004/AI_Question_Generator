from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
import os

# Set OpenAI API Key
os.environ["OPENAI_API_KEY"] = #Enter OPEN API KEY

# Load Document
loader = PyPDFLoader("CN2.pdf")
documents = loader.load()

# Split Text
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
texts = text_splitter.split_documents(documents)

# Create Embeddings
embedding_function = OpenAIEmbeddings()
vector_db = Chroma.from_documents(texts, embedding_function)

# Setup Retrieval Chain
qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(), retriever=vector_db.as_retriever())

# Generate Questions
prompt = "Generate 5 insightful questions from this document."
questions = qa.run(prompt)

print("Generated Questions from the pdf given :")
print(questions)