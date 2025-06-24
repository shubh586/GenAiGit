
from chunking import getTranscriptChunks
from loader import  extract_video_id
from vector_store import video_already_processed,create_chroma_store
from embedding import getEmbeddinModel
from langchain_chroma import Chroma 


video_id=extract_video_id("https://www.youtube.com/watch?v=XpIMuCeEtSk")

CHROMA_DIR = "./chroma_store"
COLLECTION_NAME = "youtube_video"
embedding_model = getEmbeddinModel()


if not video_already_processed(video_id):
    chunk_list=getTranscriptChunks("https://www.youtube.com/watch?v=XpIMuCeEtSk")
    anythin=create_chroma_store(chunk_list, video_id)
    print(anythin)
    print("stored sucessfully")
else:
    print("Already processed")

# def delete_video_chunks(video_id: str):
#     db = Chroma(
#         persist_directory=CHROMA_DIR,
#         embedding_function=embedding_model,
#         collection_name=COLLECTION_NAME
#     )
    
#     # Delete documents where metadata "source" is equal to the video_id
#     db.delete(where={"source": video_id})
#     print(f"Deleted all chunks with video_id: {video_id}")
# delete_video_chunks(video_id)

