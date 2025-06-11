from typing import List, Dict
import PyPDF2
import os
from langchain_redis import RedisVectorStore
from langchain_core.documents import Document
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from .redis_client import get_redis_client, get_redis_url

def get_embeddings():
    """Initializes and returns the Ollama embeddings."""
    return OllamaEmbeddings(model="nomic-embed-text", base_url=os.getenv("OLLAMA_BASE_URL", "http://ollama:11434"))

# Helper function to get project root
def get_project_root() -> Path:
    # Assumes this file is in src/ or a similar subdir
    return Path(__file__).resolve().parent.parent

class DocumentProcessor:
    def __init__(self, docs_dir: str = "documentation"):
        self.project_root = get_project_root()
        self.docs_dir = self.project_root / docs_dir
        self.embeddings = get_embeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from a PDF file."""
        text = ""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text

    def process_documents(self, session_id: str):
        """Process all PDF documents in the documentation directory for a session."""
        documents = []
        
        if not self.docs_dir.exists():
            self.docs_dir.mkdir(parents=True, exist_ok=True)
            return 0

        for filename in os.listdir(self.docs_dir):
            if filename.endswith('.pdf'):
                file_path = self.docs_dir / filename
                text = self.extract_text_from_pdf(file_path)
                
                chunks = self.text_splitter.split_text(text)
                
                for i, chunk in enumerate(chunks):
                    doc = Document(
                        page_content=chunk,
                        metadata={
                            "source": filename,
                            "chunk": i,
                            "total_chunks": len(chunks)
                        }
                    )
                    documents.append(doc)
        
        if documents:
            clear_redis_session(session_id)
            vector_store = get_vector_store(session_id)
            vector_store.add_documents(documents)
            
        return len(documents)

    def query_documentation(self, session_id: str, query: str, k: int = 3) -> List[Dict]:
        """Query the documentation for relevant information for a session."""
        vector_store = get_vector_store(session_id)
        if not vector_store:
            return []
        
        results = vector_store.similarity_search_with_score(query, k=k)
        return [
            {
                "content": doc.page_content,
                "source": doc.metadata["source"],
                "relevance_score": score,
                "chunk": doc.metadata.get("chunk"),
                "total_chunks": doc.metadata.get("total_chunks")
            }
            for doc, score in results
        ]

    def get_documentation_context(self, session_id: str, analysis_results: Dict) -> Dict:
        """Get relevant documentation context for the analysis results for a session."""
        context = {
            "error_patterns": [],
            "recommendations": [],
            "best_practices": []
        }
        
        # Query for error patterns
        error_queries = []
        for correlation in analysis_results.get("correlations", []):
            for error in correlation.get("error_chains", []):
                if isinstance(error, dict) and "message" in error:
                    error_type = error.get("error_type", "unknown")
                    if error_type == "unknown":
                        message = error["message"].lower()
                        if "timeout" in message:
                            error_type = "timeout"
                        elif "auth" in message or "credentials" in message:
                            error_type = "authentication"
                        elif "rate limit" in message:
                            error_type = "rate_limit"
                        elif "connection" in message:
                            error_type = "connection"
                        elif "queue" in message:
                            error_type = "queue"
                        elif "performance" in message or "latency" in message:
                            error_type = "performance"
                    error_queries.append(f"error pattern {error_type}")
        
        for query in set(error_queries):
            results = self.query_documentation(session_id, query)
            if results:
                context["error_patterns"].extend(results)
        
        # Query for recommendations
        recommendation_queries = [
            "best practices for handling",
            "recommended solutions for",
            "troubleshooting guide for"
        ]
        
        for query in recommendation_queries:
            results = self.query_documentation(session_id, query)
            if results:
                context["recommendations"].extend(results)
        
        # Query for best practices
        best_practice_queries = [
            "best practices for",
            "performance optimization",
            "security best practices"
        ]
        
        for query in best_practice_queries:
            results = self.query_documentation(session_id, query)
            if results:
                context["best_practices"].extend(results)
        
        return context

def clear_redis_session(session_id: str):
    """
    Clear all keys associated with a session_id in Redis.
    """
    try:
        redis_client = get_redis_client()
        # This is a bit of a hack. The python client doesn't have a direct way to delete an index.
        # We can flush the whole DB, but that's not ideal if it's shared.
        # A better approach is to delete the keys associated with the index.
        # FT.DROPINDEX is the command, but redis-py doesn't wrap it nicely for this use case.
        # We will instead delete all docs, which is sufficient for this example.
        schema_key = f"index:{session_id}"
        if redis_client.exists(schema_key):
             redis_client.ft(session_id).dropindex(delete_documents=True)
    except Exception as e:
        print(f"Could not clear Redis session {session_id}: {e}")


def get_vector_store(session_id: str):
    """
    Get the vector store for a given session ID.
    """
    redis_url = get_redis_url()
    return RedisVectorStore(
        embeddings=get_embeddings(),
        redis_url=redis_url,
        index_name=session_id,
    ) 