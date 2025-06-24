from langchain_chroma import Chroma 
from langchain_core.documents import Document
from typing import List,Optional
from embedding import getEmbeddinModel
# from YoutubeQASummary.config import CHROMA_DIR,COLLECTION_NAME
from langchain.retrievers.multi_query import MultiQueryRetriever
from prompts import multiqueryprompt


CHROMA_DIR = "./chroma_store"
COLLECTION_NAME = "youtube_video"
embedding_model = getEmbeddinModel()


def create_chroma_store(text_chunks: List[str],youtubeID:Optional[str]=None) -> Chroma:
    docs = [Document(page_content=chunk,metadata={"source": youtubeID or "youtubeID"}) for chunk in text_chunks]
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embedding_model,
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
    )
    return vectordb


def load_chroma_store() -> Chroma:
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embedding_model,
        collection_name=COLLECTION_NAME
    )



def get_retriever():
    db=load_chroma_store() 
    if db:
       return db.as_retriever()
    return None


def get_multiquery_retriver():
    baseretriver=get_retriever()
    if baseretriver is None:
        raise ValueError("Croma store is not laded properly")
    return MultiQueryRetriever.from_llm(
        retriever=baseretriver,
        llm=model,
        prompts=multiqueryprompt
    )


def video_already_processed(video_id: str) -> bool:
    db = load_chroma_store()
    if not db:
        return False

    results = db.get(where={"source": video_id})
    return len(results["ids"]) > 0






# if not video_already_processed(video_id):
#     chunks = splitter.split_text(transcript)
#     create_chroma_store(chunks, youtubeID=video_id)
# else:
#     print("Already processed")
