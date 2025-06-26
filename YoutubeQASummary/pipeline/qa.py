from model import answering_model,classification_model
from prompts import classification_prompt ,chat_prompt,answer_prompt
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from vector_store import YouTubeVectorStore
store=YouTubeVectorStore("XpIMuCeEtSk")



def classification(query):
    chain=classification_prompt|classification_model
    result=chain.invoke({"query":query})
    result=result.model_dump()
    print("result query answer ",result["is_topic_query"])
    return result["is_topic_query"]

def answer(query: str, retriever):
    topic_query = classification(query)
    if topic_query=="NO":
        chain=chat_prompt | answering_model
        result=chain.invoke({"query":query})
        return AIMessage(content=result.content.strip())

    retrieve_dcos=retriever.invoke(query)[:2] #bkl return type List[Document] hai
    if not retrieve_dcos:
        return AIMessage(content="No relevant content found in the transcript.")

    context="\n\n".join(doc.page_content for doc in retrieve_dcos)
    chain=answer_prompt|answering_model
    result=chain.invoke({"query":query,"context":context})
    return AIMessage(content=result.content.strip())


#testing chalu ho gaya re baba :

if store.video_already_processed():
    retriever = store.get_multiquery_retriever()
    chat_history = []
    print("🔁 Type your question about the video (type 'exit' to quit)\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            break
        chat_history.append(HumanMessage(content=user_input))
        response = answer(user_input, retriever)
        print("AI:", response.content)
        chat_history.append(response)
    print("\n👋 Exiting chat.")


    
    

    



