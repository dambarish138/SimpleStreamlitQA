import streamlit as st
import os
import tempfile
import shutil
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from streamlit_chat import message
from langchain_chroma import Chroma
from langchain_community.callbacks.manager import get_openai_callback

embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key="<API Key>")

def run_app():
    load_dotenv()
    st.set_page_config(page_title="Lockton Demo")
    st.header("Lockton Demo")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    with st.sidebar:
        uploaded_files =  st.file_uploader("Upload your file",type=['pdf'],accept_multiple_files=False)
        process = st.button("Process")
    if process:
        files_text = get_pdf_text(uploaded_files)
        text_chunks = get_text_chunks(files_text)
        vetorestore = get_vectorstore(text_chunks)
     
        st.session_state.conversation = get_conversation_chain(vetorestore) 

        st.session_state.processComplete = True

    if  st.session_state.processComplete == True:
        user_question = st.chat_input("Chat with your file")
        if user_question:
            handel_userinput(user_question)


def get_files_text(uploaded_files):
    text = ""
    for uploaded_file in uploaded_files:
        split_tup = os.path.splitext(uploaded_file.name)
        file_extension = split_tup[1]
        if file_extension == ".pdf":
            text += get_pdf_text(uploaded_file)
        else:
            print("Wrong file uploaded")
    return text


def get_pdf_text(pdf):
    if pdf is not None:
        # save as tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
            shutil.copyfileobj(pdf, tmpfile)
            file_path = tmpfile.name
    loader = PyMuPDFLoader(file_path)
    data = loader.load()
    print("Data Read Complete")
    return data

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=900,
        chunk_overlap=100,
        length_function=len
    )
    docs = text_splitter.split_documents(text)
    return docs

def get_vectorstore(text_chunks):
    knowledge_base = Chroma.from_documents(text_chunks, embedding=embeddings)
    return knowledge_base

def get_conversation_chain(vetorestore):
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.3,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key="<API Key>", 
)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vetorestore.as_retriever(),
        memory=memory
    )
    return conversation_chain

# This function takes a user question as input, sends it to a conversation model and displays the conversation history along with some additional information.
def handel_userinput(user_question):

    with get_openai_callback() as cb:
        response = (st.session_state.conversation).invoke({'question':user_question})
    st.session_state.chat_history = response['chat_history']

    response_container = st.container()

    with response_container:
        for i, messages in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                message(messages.content, is_user=True, key=str(i))
            else:
                message(messages.content, key=str(i))
        st.write(f"Total Tokens: {cb.total_tokens}" f", Prompt Tokens: {cb.prompt_tokens}" f", Completion Tokens: {cb.completion_tokens}" f", Total Cost (USD): ${cb.total_cost}")


run_app()
