from typing import List, Dict
import PyPDF2
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from pathlib import Path

# Helper function to get project root
def get_project_root() -> Path:
    # Assumes this file is in src/ or a similar subdir
    return Path(__file__).resolve().parent.parent

class DocumentProcessor:
    def __init__(self, docs_dir: str = "documentation"):
        self.project_root = get_project_root()
        self.docs_dir = self.project_root / docs_dir
        self.embeddings = OllamaEmbeddings(model="llama3.2:1b")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        self.vector_store = None

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from a PDF file."""
        text = ""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text

    def process_documents(self):
        """Process all PDF documents in the documentation directory."""
        documents = []
        
        if not self.docs_dir.exists():
            self.docs_dir.mkdir(parents=True, exist_ok=True)
            # No documents to process if directory was just created
            return 0

        for filename in os.listdir(self.docs_dir):
            if filename.endswith('.pdf'):
                file_path = self.docs_dir / filename
                text = self.extract_text_from_pdf(file_path)
                
                # Split text into chunks
                chunks = self.text_splitter.split_text(text)
                
                # Create documents with metadata
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
        
        # Create vector store
        if documents:
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
        return len(documents)

    def query_documentation(self, query: str, k: int = 3) -> List[Dict]:
        """Query the documentation for relevant information."""
        if not self.vector_store:
            return []
        
        results = self.vector_store.similarity_search_with_score(query, k=k)
        return [
            {
                "content": doc.page_content,
                "source": doc.metadata["source"],
                "relevance_score": score,
                "chunk": doc.metadata["chunk"],
                "total_chunks": doc.metadata["total_chunks"]
            }
            for doc, score in results
        ]

    def get_documentation_context(self, analysis_results: Dict) -> Dict:
        """Get relevant documentation context for the analysis results."""
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
                    # Extract error type from the message if not directly available
                    error_type = error.get("error_type", "unknown")
                    if error_type == "unknown":
                        # Try to infer error type from message
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
        
        for query in error_queries:
            results = self.query_documentation(query)
            if results:
                context["error_patterns"].extend(results)
        
        # Query for recommendations
        recommendation_queries = [
            "best practices for handling",
            "recommended solutions for",
            "troubleshooting guide for"
        ]
        
        for query in recommendation_queries:
            results = self.query_documentation(query)
            if results:
                context["recommendations"].extend(results)
        
        # Query for best practices
        best_practice_queries = [
            "best practices for",
            "performance optimization",
            "security best practices"
        ]
        
        for query in best_practice_queries:
            results = self.query_documentation(query)
            if results:
                context["best_practices"].extend(results)
        
        return context 