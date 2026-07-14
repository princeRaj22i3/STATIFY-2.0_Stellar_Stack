from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from mcp.server.fastmcp import FastMCP

from schema3 import RetrievalResponse


# ==========================================================
# Global Variables
# ==========================================================

retriever = None
vector_db = None
embedding_model = None

BASE_DIR = Path(__file__).resolve().parent

CHROMA_DIR = BASE_DIR / "chroma_db"

PDF_PATH = (
    BASE_DIR
    / "data"
    / "Module 2_Technical Analysis.pdf"
)

mcp = FastMCP("Statify Vector DB")


# ==========================================================
# PDF Loader
# ==========================================================

def load_pdf(pdf_path: str | Path):

    loader = PyPDFLoader(str(pdf_path))

    return loader.load()


# ==========================================================
# Text Splitter
# ==========================================================

def split_documents(documents):

    splitter = RecursiveCharacterTextSplitter(

        chunk_size=500,

        chunk_overlap=50

    )

    return splitter.split_documents(documents)


# ==========================================================
# Embedding Model
# ==========================================================

def create_embedding_model():

    return HuggingFaceEmbeddings(

        model_name="sentence-transformers/all-MiniLM-L6-v2"

    )


# ==========================================================
# Chroma Database
# ==========================================================

def create_vector_database(chunks, embedding):

    if CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir()):

        db = Chroma(

            persist_directory=str(CHROMA_DIR),

            embedding_function=embedding

        )

    else:

        db = Chroma.from_documents(

            documents=chunks,

            embedding=embedding,

            persist_directory=str(CHROMA_DIR)

        )

    return db


# ==========================================================
# Initialize RAG
# ==========================================================

def initialize_rag(k: int = 3):

    global retriever
    global vector_db
    global embedding_model

    if retriever is not None:

        return

    documents = load_pdf(PDF_PATH)

    chunks = split_documents(documents)

    embedding_model = create_embedding_model()

    vector_db = create_vector_database(

        chunks,

        embedding_model

    )

    retriever = vector_db.as_retriever(

        search_type="similarity",

        search_kwargs={"k": k}

    )


# ==========================================================
# MCP Tool
# ==========================================================

@mcp.tool()

def retrieve_context(query: str) -> str:
    """
    Retrieve technical analysis context
    from the vector database.
    """

    if retriever is None:

        raise RuntimeError(

            "Retriever not initialized."

        )

    docs = retriever.invoke(query)

    context = "\n\n".join(

        doc.page_content

        for doc in docs

    )

    return RetrievalResponse(

        context=context

    ).context


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    try:

        print("Initializing Vector Database...")

        initialize_rag()

        print("Vector Database Ready.")

        print("Starting MCP Server...")

        mcp.run(
           transport="streamable-http"
        )

    except Exception as e:

        print(f"Server startup failed : {e}")