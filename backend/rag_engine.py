import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

load_dotenv()

_embeddings_cache = None

def get_embeddings():
    global _embeddings_cache
    if _embeddings_cache is None:
        _embeddings_cache = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return _embeddings_cache

def format_docs(docs):
    if not docs:
        return "No relevant documents found."
    return "\n\n".join(doc.page_content for doc in docs)

def get_answer_from_gemini(user_query: str, chat_history: list = None) -> str:
    # 1. Initialize history
    if chat_history is None:
        chat_history = []
    
    if not user_query or not user_query.strip():
        return "Please provide a valid question."

    formatted_history = []
    for msg in chat_history:
        if isinstance(msg, dict):
            if msg.get("role") == "user":
                formatted_history.append(HumanMessage(content=msg["content"]))
            elif msg.get("role") == "assistant":
                formatted_history.append(AIMessage(content=msg["content"]))
        else:
            formatted_history.append(msg)

    try:
        embeddings = get_embeddings()
        vectordb = Chroma(
            persist_directory=os.getenv("CHROMA_DB_PATH", "chroma_db"), 
            embedding_function=embeddings
        )
        retriever = vectordb.as_retriever(search_kwargs={"k": 5})

        model = ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash-lite"), 
            temperature=0,
            max_tokens=1024
        )

        condense_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        condense_q_prompt = ChatPromptTemplate.from_messages([
            ("system", condense_q_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{user_query}"),
        ])
        def contextualize_question(input_dict):
            if input_dict.get("chat_history"):
                chain = condense_q_prompt | model | StrOutputParser()
                return chain.invoke(input_dict)
            return input_dict["user_query"]
        qa_system_prompt = """You are a strict, highly accurate GitLab assistant. 
        
        CRITICAL INSTRUCTION: You must answer the user's question relying EXCLUSIVELY on the Context provided below. You are strictly forbidden from using any outside factual knowledge, internet training data, or assumptions.
        
        If the user asks you to format, shorten, or summarize previous information (e.g., "be concise", "make it shorter"), you must do so using ONLY the provided Context.
        
        If the Context below does not contain the facts needed to address the user's core topic, you MUST reply with this exact phrase and nothing else: "I don't know based on the provided documentation."
        
        Context:
        {context}"""
        
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{user_query}"),
        ])
        
        rag_chain = (
            RunnablePassthrough.assign(
                context=RunnableLambda(contextualize_question) | retriever | format_docs
            )
            | qa_prompt
            | model
            | StrOutputParser()
        )
        
        response = rag_chain.invoke({
            "user_query": user_query,
            "chat_history": formatted_history
        })
        
        return response
    
    except Exception as e:
        return f"Error processing your question: {str(e)}"

if __name__ == "__main__":
    test_history = []
    while True:
        user_input = input("Enter your question (or 'quit' to exit): ")
        if user_input.lower() == "quit":
            break
        answer = get_answer_from_gemini(user_input, test_history)
        test_history.append(HumanMessage(content=user_input))
        test_history.append(AIMessage(content=answer))
        print(f"Assistant: {answer}\n")
