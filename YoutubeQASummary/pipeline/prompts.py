
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import PromptTemplate

multiqueryprompt = PromptTemplate(
    template=(
        "You are an expert assistant helping to retrieve relevant transcript parts from a video.\n"
        "Given the following user question, generate 3 diverse and semantically distinct rephrasings "
        "that might help retrieve related content from the transcript.\n\n"
        "Original Question: {question}\n\n"
        "Provide the rephrased queries as a numbered list:\n"
        "1."
    ),
    input_variables=["question"]
)
