from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List,Optional
from loader import getTranscript


def getTranscriptChunks(url: str) -> Optional[List[str]]:
    text = getTranscript(url)
    if text:
        splitter = RecursiveCharacterTextSplitter(chunk_overlap=500, chunk_size=3000)
        return splitter.split_text(text)
    else:
        print('didnt get text')
    return None



