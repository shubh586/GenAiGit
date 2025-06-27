from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List,Optional
from loader import getTranscript
from transformers import AutoTokenizer


chunkinfrefmodel="NousResearch/Llama-2-7b-chat-hf"
def getTranscriptChunks(url: str) -> Optional[List[str]]:
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    text = getTranscript(url)
    if text:
        splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer=tokenizer ,chunk_overlap=300, chunk_size=2000
        )
        return splitter.split_text(text)
    else:
        print('didnt get text')
    return None



