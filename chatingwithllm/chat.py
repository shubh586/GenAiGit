from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

class chatAiSchema(BaseModel):
    AI: str = Field(description="give me the friedly message starting with the AI")

load_dotenv()

model = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7,
    max_retries=2,
)

chatprompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="system: you are general chatbot my friend at my college\n Always return a json object Starting with the AI:<Your ans here>"
    "Respond only with valid JSON."),
    MessagesPlaceholder(variable_name="messages")
])

parser = PydanticOutputParser(pydantic_object=chatAiSchema)

chain = chatprompt | model | parser

chatHistory = []

while True:
    userinput = input("You: ")
    if userinput.lower() == 'exit':
        break
    chatHistory.append(HumanMessage(content=userinput))
    result = chain.invoke({"messages": chatHistory})
    print("AI :",result.AI)
    chatHistory.append(AIMessage(content=result.AI))
