from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv
import os
load_dotenv()

def getEmbeddinModel():
    embedding_model= HuggingFaceEndpointEmbeddings(
    model="Snowflake/snowflake-arctic-embed-l-v2.0",
    task="feature-extraction",
    huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_KEY")
    )
    return embedding_model


