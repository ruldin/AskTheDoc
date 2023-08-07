import streamlit as st
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

def generate_response(uploaded_file, openai_api_key, query_text):
    # Load document
    if uploaded_file is not None:
        #documents = [uploaded_file.read().decode('utf-8')]
       # documents = [uploaded_file.read().decode('utf-8').encode('ascii', 'ignore').decode('utf-8')]
        documents = [uploaded_file.read().decode('utf-8','ignore')]



      #split document into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000,chunk_overlap =0)
        texts = text_splitter.create_documents(documents)
        #select embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        #create vector from document
        db = Chroma.from_documents(texts, embeddings)
        #create retriever interface
        retriever = db.as_retriever()
        #create QA chain
        #qa = RetrievelQA.from_chain_type(llm=OpenAI(openai_api_key=openai_api_key), retriever=retriever)
        qa = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key=openai_api_key), chain_type='stuff', retriever=retriever)
        #run query
        return qa.run(query_text)

#page title
st.set_page_config(page_title='📎🖇️ ask to Document', page_icon='📎', layout='wide', initial_sidebar_state='auto')
st.page_title = '📎🖇️ ask to Document'

#file upload
uploaded_file = st.file_uploader('Upload a file or article', type=['txt', 'pdf', 'docx'])

#query text
query_text = st.text_input('Enter your question:', placeholder = 'Please provide a short summary.', disabled=not uploaded_file)

#form input and query
result = []
with st.form('my_form',clear_on_submit=True):
    openai_api_key = st.text_input('OpenAI API Key', type='password', disabled=not (uploaded_file and query_text))
    submitted = st.form_submit_button('Submit', disabled=False) #not (uploaded_file and query_text and openai_api_key))
    if submitted and openai_api_key.startswith('sk-'):
        with st.spinner('Generating response...'):
            response = generate_response(uploaded_file, openai_api_key, query_text)
            result.append(response)
            del openai_api_key

if len(result):
    st.info('Answer:'.join(response))
    st.write('Otro:'.join(result[0]))


