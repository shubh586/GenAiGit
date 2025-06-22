from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnableParallel, RunnablePassthrough, RunnableSequence, RunnableMap
from langchain_core.output_parsers import StrOutputParser



load_dotenv()
model = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.0,
    max_retries=2,
    # other params...
)

report_template = PromptTemplate(
    input_variables=["topic"],
    template=(
        "You are an experienced environmental researcher.\n"
        "Write a comprehensive, in-depth research paper on the topic: {topic}.\n"
        "Ensure the paper includes:\n"
        "- A detailed introduction and background\n"
        "- Current trends and data analysis\n"
        "- Challenges and environmental implications\n"
        "- Real-world case studies or examples\n"
        "- Conclusions and future recommendations\n"
        "Use scientific tone and cite reliable sources if possible."
    )
)


report_summarizer = PromptTemplate(
    input_variables=["text"],
    template=(
        "You are a professional research analyst and summarizer.\n"
        "Read the following research paper and provide a clear, concise summary highlighting the key points, "
        "main arguments, findings, and conclusions:\n\n{text}\n\n"
        "Ensure the summary is informative and easy to understand."
    )
)


parser=StrOutputParser()



report_chain=report_template|model|parser
summary_chain=report_summarizer|model|parser



summary_plus_report= report_chain|RunnableParallel({
    "report":RunnablePassthrough(),
    "summary":RunnableMap({"text":RunnableLambda(lambda x: x)})|summary_chain
})

result=summary_plus_report.invoke({"topic":"soil pollution"})
print("==============================Report====================\n")
print(result['report'])
print("\n==============================Summary====================")
print(result['summary'])