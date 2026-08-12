from langchain_core.documents import Document

def pinecone_results_structured(results):
    """It converts pinecone results into a structured results so that grader function can easily understand 
    and read the documents clearly, therefore we structuring it as List[Documents]"""
    documents = []

    for match in results.matches:
        metadata = match.metadata or {}

        code_text = metadata.get("text","")
        file_path = metadata.get('file_path',"unknown_path")
        doc_index = metadata.get("doc_index")

        doc = Document(
            page_content=code_text,
            metadata={
                "id":match.id,
                "file_path":file_path,
                "score":match.score,
                "doc_index":doc_index,
            }
        )

        documents.append(doc)

    return documents