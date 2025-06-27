import asyncio
from typing import List
from langchain_core.messages import HumanMessage
from prompts import summarizer_prompt,final_summary_prompt
from langchain_core.output_parsers import StrOutputParser
from chunking import getTranscriptChunks
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from transformers import AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import time
from model import chunk_model, summary_model
from loader import extract_video_id
from vector_store import YouTubeVectorStore
load_dotenv()
async def summarize_chunk(chunk: str) -> str:
    """Summarize a single text chunk with throttling.""" 
    summary_chain=chunk_model.get_chain(summarizer_prompt)
    response = await summary_chain.ainvoke({"text":chunk})
    return response.content

async def summarize_summary_chunk(chunk: str) -> str:
    """Summarize a single text chunk with throttling.""" 
    summary_chain=summary_model.get_chain(final_summary_prompt)
    response = await summary_chain.ainvoke({"summary_input":chunk})
    return response.content

def summarize_chunks(chunks: List[str]) -> List[str]:
    """Summarize all chunks in parallel with throttling."""
    async def run_all():
        results = []
        rpm=chunk_model.get_rpm()
        for i in range(0, len(chunks), rpm):
            batch = chunks[i:i+rpm]
            batch_tasks = [summarize_chunk(c) for c in batch]
            batch_results = await asyncio.gather(*batch_tasks)
            results.extend(batch_results)
            if i + rpm < len(chunks):
                await asyncio.sleep(60)  
        return results
    return asyncio.run(run_all())

def summarize_all(chunk_summaries: List[str]) -> str:
    """Merge and summarize all chunk-level summaries."""
    summary_input = "\n".join(chunk_summaries)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(    
        tokenizer=tokenizer,
        chunk_overlap=200,
        chunk_size=5000
    )
    chunked_summary_input=splitter.split_text(summary_input)
    """Summarize all chunks in parallel with throttling."""
    async def run_all():
        results = []
        rpm=summary_model.get_rpm()
        for i in range(0, len(chunked_summary_input), rpm):
            batch = chunked_summary_input[i:i+rpm]
            batch_tasks = [summarize_summary_chunk(c) for c in batch]
            batch_results = await asyncio.gather(*batch_tasks)
            results.extend(batch_results)
            if i + rpm < len(chunked_summary_input):
                await asyncio.sleep(60)  
        return "\n".join(results)
    return asyncio.run(run_all())
  

VIDEO="https://www.youtube.com/watch?v=q9icMJ48z6U"
VIDEO2="https://www.youtube.com/watch?v=XpIMuCeEtSk"

if __name__ == "__main__":
    chunk_list = getTranscriptChunks(VIDEO2)
    print("\nThe length of the chunk list is:", len(chunk_list))
    start_time = time.time()
    result_list = summarize_chunks(chunk_list)
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"\n🕒 Time taken to summarize all chunks: {elapsed_time:.2f} seconds")
    print("\nThe length of the result list is:", len(result_list))
    vid=extract_video_id(VIDEO2)
    store=YouTubeVectorStore(vid)
    print("\n\nhere is the final summary \n", summarize_all(result_list))

