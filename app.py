import streamlit as st
from dotenv import load_dotenv 
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS  # allows to store embeddings (in this case in local area, and will erase when applicaiton is closed)
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
from htmlTemplates import css, bot_template, user_template
# from haystack.retriever.dense import FAISSRetriever
# from haystack.document_store.faiss import FAISSDocumentStore
# from haystack.retriever.base import BaseRetriever


def get_pdf_text(pdf_docs):
    '''
    This funciton gets reads the pdfs and returns a combined text blob
    '''
    text = "" ## variable which will contain all of the raw text of from pdfs
    for pdf in pdf_docs: 
        pdf_reader = PdfReader(pdf) ## create a pdf reader object ehich has pages from the uploaded pdfs
        for page in pdf_reader.pages: ## loop through pages
            text += page.extract_text() ## extract text and add it to the variable
    return text


def get_text_chunks(text):
    '''
    This function divides the text extracted from pdfs into chunks
    '''
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200, ## When the chunk size runs out in the middle of a chunk, then in the next chunk this will ensure to read from 200 characters before, so that the meaning of the chunk/sentense is not lost due to icorrect selection of chunks
        length_function=len
    )

    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name="sentence-transformers/msmarco-distilroberta-base-v2")
    ## retriever = FAISSRetriever(texts=text_chunks, embeddings=embeddings)
    ## vectorstore = FAISSDocumentStore(retriever=retriever)
    vectorstore = FAISS.from_texts(texts=text_chunks,embedding=embeddings) # Generating the database with text chunks and embeddings
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    load_dotenv() #we will run this function in order to be able to use variables from env files
    st.set_page_config(page_title="Chat with multipel pdfs", page_icon=":books:")

    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None


    st.header("Chat with multiple pdfs :books:")
    user_question = st.text_input("Ask a question about your document")
    if user_question:
        handle_userinput(user_question)


    # sidebar to upload docs
    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs=st.file_uploader(
            "Upload your pdfs here and click on  'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"): # All the content in this loop will be processed while the user sees a spnning wheel
                
                # getting the pdfs
                raw_text = get_pdf_text(pdf_docs)
                #st.write(raw_text) #testing if text blob returns

                # getting the text chunks
                text_chunks = get_text_chunks(raw_text)
                #st.write(text_chunks)

                # creating vector store to be able to store embeddings
                vectorstore = get_vectorstore(text_chunks)

                # Create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)

if __name__ =='__main__':
    main()