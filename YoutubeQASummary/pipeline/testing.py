
from chunking import getTranscriptChunks
from loader import  extract_video_id
from vector_store import YouTubeVectorStore 
import numpy as np

# https://www.youtube.com/watch?v=q9icMJ48z6U  elon wali link
# https://www.youtube.com/watch?v=XpIMuCeEtSk
if __name__ ==  "__main__":
    VIDEO="https://www.youtube.com/watch?v=q9icMJ48z6U"
    video_id=extract_video_id(VIDEO)
    store=YouTubeVectorStore(video_id)
    db = store.load_faiss_store()
    
    if not store.video_already_processed():
        chunk_list=getTranscriptChunks(VIDEO)
        anythin=store.create_faiss_store(chunk_list)
        print(anythin)
        print("stored sucessfully")
    else:
        print("Already processed")
        results = store.get_chunks()
        print(results)
    
        index = db.index  # FAISS index object
        embedding_vector = index.reconstruct(0)
        print("Embedding vector shape:", embedding_vector.shape)
        print("First 10 dimensions of embedding vector:\n", embedding_vector    [:10]) 
    
    
    
    