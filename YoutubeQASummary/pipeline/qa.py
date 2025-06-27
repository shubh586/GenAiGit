from model import answering_model,classification_model
from prompts import classification_prompt ,chat_prompt,answer_prompt
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from vector_store import YouTubeVectorStore
from summarizer import summarize_all
from typing import List


def classification(query):
    chain=classification_prompt|classification_model
    result=chain.invoke({"query":query})
    result=result.model_dump()
    #print("result query answer ",result["is_topic_query"])
    return result["is_topic_query"]

def answer(query: str,chathistory:List[str], retriever):
    topic_query = classification(query)
    if topic_query=="NO":
        chain=chat_prompt | answering_model
        result=chain.invoke({"query":query,"chathistory":chathistory})
        return AIMessage(content=result.content.strip())

    retrieve_dcos=retriever.invoke(query)[:2] #bkl return type List[Document] hai
    if not retrieve_dcos:
        return AIMessage(content="No relevant content found in the transcript.")

    context="\n\n".join(doc.page_content for doc in retrieve_dcos)
    chain=answer_prompt|answering_model
    result=chain.invoke({"query":query,"context":context})
    return AIMessage(content=result.content.strip())



if __name__ ==  "__main__":

    store=YouTubeVectorStore("XpIMuCeEtSk")
    if store.video_already_processed():
        print("we are all set to go ...\n\n")
        print("Here is the summary the video :\n")
        summary_chunk_list=store.get_chunks()
        summary=summarize_all(summary_chunk_list)
        print(summary)
        retriever = store.get_multiquery_retriever()
        chat_history = []
        print("\n\n🔁 Type your question about the video (type 'exit' to quit)\n")

        while True:
            user_input = input("You: ").strip()
            if user_input.lower() in ["exit", "quit"]:
                break
            chat_history.append(HumanMessage(content=user_input))
            response = answer(user_input,chat_history, retriever)
            print("\nAI:", response.content,"\n")
            chat_history.append(response)
        print("\n👋 Exiting chat.")









