import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()


st.set_page_config(page_title="Smart DSA Tutor", layout="wide")


st.markdown("""
    <style>
    /* Hide the Streamlit main menu and footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    /* Hide the 'Deploy' button specifically */
    .stDeployButton {display:none;}
    
    /* Professional input box spacing */
    .stChatInputContainer {
        padding-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

with st.sidebar:
    st.title("System Configuration")
    


st.title("Smart DSA Tutor")
st.markdown("##### *Intelligent Edge-AI Learning Platform*")
st.divider()

@st.cache_resource
def init_tutor():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    llm = ChatOllama(model="llama3.2", temperature=0.3) 
    index_name = "dsa-tutor-local" 
    
    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    
    return llm, retriever

llm, retriever = init_tutor()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


for msg in st.session_state.chat_history:
    
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_query = st.chat_input("Enter your query regarding Data Structures and Algorithms...")

if user_query:
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing computational context..."):
            
            retrieved_docs = retriever.invoke(user_query)
            context = "\n\n".join([d.page_content for d in retrieved_docs])
            
         
            system_instr = SystemMessage(content="""You are a strict, but helpful Socratic DSA Tutor. Always reply in soft tone. Do not use strong language to answer the student. 
RULES:
1. Always reply in the EXACT language the user uses (Hindi/English/Hinglish).
2. Never give the direct solution immediately. Ask guiding questions.
3. Use the provided context to help the student.
4. Never give long replies.
5. Never provide code just help user in writing it themselves.""")
            
            messages = [system_instr]
            for chat in st.session_state.chat_history[:-1]: 
                if chat["role"] == "user":
                    messages.append(HumanMessage(content=chat["content"]))
                else:
                    messages.append(AIMessage(content=chat["content"]))
            
            final_input = f"Context from Notes:\n{context}\n\nStudent: {user_query}"
            messages.append(HumanMessage(content=final_input))
            
         
            response = llm.invoke(messages)
            ai_answer = response.content
            
 
            st.markdown(ai_answer)
            st.session_state.chat_history.append({"role": "assistant", "content": ai_answer})