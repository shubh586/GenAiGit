
import os
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from typing import List, Optional
from embedding import getEmbeddinModel
from langchain.retrievers.multi_query import MultiQueryRetriever
from prompts import multiqueryprompt
import pickle

CHROMA_DIR = "/home/shubh/Programming/genai/Mywork/llm/YoutubeQASummary/data/"
COLLECTION_NAME = "youtube_video"

class YouTubeVectorStore:
    def __init__(self):
        self.embedding_model = getEmbeddinModel()
        self.index_dir = os.path.join(CHROMA_DIR, COLLECTION_NAME)
        self.vectorstore = self.load_faiss_store()

    def create_faiss_store(self, text_chunks: List[str], youtubeID: Optional[str] = None):
        docs = [Document(page_content=chunk, metadata={"source": youtubeID or "youtubeID"}) for chunk in text_chunks]
        faiss_store = FAISS.from_documents(documents=docs, embedding=self.embedding_model)
        os.makedirs(self.index_dir, exist_ok=True)
        faiss_store.save_local(self.index_dir)
        self.vectorstore = faiss_store
        return faiss_store

    def load_faiss_store(self) -> Optional[FAISS]:
        if os.path.exists(os.path.join(self.index_dir, "index.faiss")):
            return FAISS.load_local(self.index_dir, self.embedding_model,allow_dangerous_deserialization=True )
        return None

    def get_retriever(self):
        if not self.vectorstore:
            raise ValueError("FAISS store is not loaded.")
        return self.vectorstore.as_retriever()

    def get_multiquery_retriever(self, model):
        base_retriever = self.get_retriever()
        return MultiQueryRetriever.from_llm(
            retriever=base_retriever,
            llm=model,
            prompts=multiqueryprompt
        )

    def video_already_processed(self, video_id: str) -> bool:
        if not self.vectorstore:
            return False
        results = self.vectorstore.similarity_search("what elon musk said", k=5, filter={"source": video_id})
        return len(results) > 0

    def get_chunks(self, video_id: str):
        if not self.vectorstore:
            return []
        return self.vectorstore.similarity_search("what elon musk said", k=5, filter={"source": video_id})




# from langchain_chroma import Chroma 
# from langchain_core.documents import Document
# from typing import List,Optional
# from embedding import getEmbeddinModel
# from langchain.retrievers.multi_query import MultiQueryRetriever
# from prompts import multiqueryprompt
# import chromadb
# from chromadb.utils import embedding_functions

# CHROMA_DIR = "/home/shubh/Programming/genai/Mywork/llm/YoutubeQASummary/data"
# COLLECTION_NAME = "youtube_video"

# class  YouTubeVectorStore:
#     def __init__(self):
#         self.embedding_model = getEmbeddinModel()
#         self.chroma_dir = CHROMA_DIR
#         self.collection_name = COLLECTION_NAME


#     def create_chroma_store(self, text_chunks: List[str], youtubeID: Optional[str] = None) -> Chroma:
#         docs = [
#             Document(page_content=chunk, metadata={"source": youtubeID or "youtubeID"})
#             for chunk in text_chunks
#         ]
#         embeddings = self.embedding_model.embed_documents(text_chunks)  
#         print(embeddings)
#         vectordb = Chroma.from_documents(
#              documents=docs,
#              embedding=self.embedding_model,  
#              persist_directory=self.CHROMA_DIR,
#              collection_name=self.COLLECTION_NAME
#          )

#     def load_chroma_store(self) -> Chroma:
#         return Chroma(
#             embedding_function=self.embedding_model,
#             persist_directory=self.chroma_dir,
#             collection_name=self.collection_name,
#         )



#     def get_retriever(self):
#         db=self.load_chroma_store() 
#         if db:
#            return db.as_retriever()
#         return None


#     def get_multiquery_retriver(self):
#         baseretriver=self.get_retriever()
#         if baseretriver is None:
#             raise ValueError("Croma store is not laded properly")
#         return MultiQueryRetriever.from_llm(
#             retriever=baseretriver,
#             llm=model,
#             # model will come here
#             prompts=multiqueryprompt
#         )


#     def video_already_processed(self,video_id: str) -> bool:
#         db = self.load_chroma_store()
#         if not db:
#             return False

#         results = db.get(where={"source": video_id})
#         return len(results["ids"]) > 0


#     def get_chunks(self,video_id: str):
#         db = self.load_chroma_store()
#         return  db.get(where={"source": video_id})


#     def delete_video_chunks(self,video_id: str):
#         db = self.load_chroma_store()
#         db.delete(where={"source": video_id})
#         print(f"Deleted all chunks with video_id: {video_id}")
