import asyncio
from typing import List
from langchain_core.messages import HumanMessage
from prompts import summarizer_prompt
from langchain_core.output_parsers import StrOutputParser
from chunking import getTranscriptChunks
from langchain_groq import ChatGroq
from dotenv import load_dotenv


MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct"
MAX_TOKENS_PER_MINUTE = 30000
TOKENS_PER_INPUT_CHUNK = 3000  # Your chunk size
MAX_TOKEN_PER_OUTPUT=(TOKENS_PER_INPUT_CHUNK//3 ) + 200  
TOKENS_PER_REQUEST=TOKENS_PER_INPUT_CHUNK + MAX_TOKEN_PER_OUTPUT
REQUESTS_PER_MINUTE = MAX_TOKENS_PER_MINUTE // TOKENS_PER_REQUEST  



VIDEO="https://www.youtube.com/watch?v=q9icMJ48z6U"
VIDEO2="https://www.youtube.com/watch?v=XpIMuCeEtSk"
load_dotenv()
model2="llama-3.3-70b-versatile"
model1="meta-llama/llama-4-scout-17b-16e-instruct"
model = ChatGroq(
    model=model1,
    temperature=0.3,
    max_retries=2,
)


async def summarize_chunk(chunk: str) -> str:
    """Summarize a single text chunk with throttling.""" 
    summary_chain=summarizer_prompt|model
    response = await summary_chain.ainvoke({"text":chunk})
    return response.content


def summarize_chunks(chunks: List[str]) -> List[str]:
    """Summarize all chunks in parallel with throttling."""
    async def run_all():
        results = []
        for i in range(0, len(chunks), REQUESTS_PER_MINUTE):
            batch = chunks[i:i+REQUESTS_PER_MINUTE]
            batch_tasks = [summarize_chunk(c) for c in batch]
            batch_results = await asyncio.gather(*batch_tasks)
            results.extend(batch_results)
            if i + REQUESTS_PER_MINUTE < len(chunks):
                await asyncio.sleep(60)  
        return results
    return asyncio.run(run_all())


def summarize_all(chunk_summaries: List[str]) -> str:
    """Merge and summarize all chunk-level summaries."""
    summary_input = "\n".join(chunk_summaries)
    final_prompt = HumanMessage(content=f"Summarize the full video based on these parts:\n{summary_input}")
    response = model.invoke([final_prompt])
    return response.content



chunk_list = getTranscriptChunks(VIDEO2)
print("\nThe length of the chunk list is:", len(chunk_list))
print("The length of the second-to-last chunk is:", len(chunk_list[-2]))
print("Second-to-last chunk:\n", chunk_list[-2])

import time
start_time = time.time()
result_list = summarize_chunks(chunk_list)
end_time = time.time()
elapsed_time = end_time - start_time

print(f"\n🕒 Time taken to summarize all chunks: {elapsed_time:.2f} seconds")

print("\nThe length of the result list is:", len(result_list))
print("The length of the second-to-last result is:", len(result_list[-2]))
print("Second-to-last summary result:\n", result_list[-2])

