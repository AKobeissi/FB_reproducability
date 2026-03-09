from .rag_dependencies import (
    RecursiveCharacterTextSplitter, 
    HuggingFaceEmbeddings, 
    BM25Retriever
)


def __getattr__(name: str):
    if name == "RAGExperiment":
        from .rag_experiments import RAGExperiment
        return RAGExperiment
    raise AttributeError(f"module 'src.core' has no attribute '{name}'")


__all__ = ["RAGExperiment", "RecursiveCharacterTextSplitter", "HuggingFaceEmbeddings", "BM25Retriever"]