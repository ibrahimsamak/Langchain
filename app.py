import streamlit as st
import openai
import tiktoken
import time
import tempfile

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

from youtube_transcript_api import YouTubeTranscriptApi

pages = None
st.title("LangChain ile özetlenmesi veya sorular ve cevaplar")
pdf_file = st.file_uploader("PDF file", type="pdf")
llm = OpenAI(model='gpt-3.5-turbo-instruct', openai_api_key='sk-proj-ANmYN7yLbsp0WeQaY0GUT3BlbkFJQ59X69TRoQKIOPSqs15P', temperature=0)
openai.api_key = 'sk-proj-ANmYN7yLbsp0WeQaY0GUT3BlbkFJQ59X69TRoQKIOPSqs15P'; 
page_selection = st.radio("", ["Özet", "Soru", "Youtube", "Website"])

text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 1400,
    chunk_overlap  = 200,
    length_function = len,
)

def get_transcript(youtube_url):
    video_id = youtube_url.split("v=")[-1]
    transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
    try:
        transcript = transcript_list.find_manually_created_transcript()
        language_code = transcript.language_code  
    except:
        try:
            generated_transcripts = [trans for trans in transcript_list if trans.is_generated]
            transcript = generated_transcripts[0]
            language_code = transcript.language_code
        except:
            raise Exception("hata")

    full_transcript = " ".join([part['text'] for part in transcript.fetch()])
    return full_transcript, language_code

def summarize_with_langchain_and_openai(transcript, language_code, model_name='gpt-3.5-turbo'):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
    texts = text_splitter.split_text(transcript)
    text_to_summarize = " ".join(texts[:4]) 

    system_prompt = 'I want you to act as a Life Coach that can create good summaries!'
    prompt = f'''Summarize the following text in {language_code}.
    Text: {text_to_summarize}

    Add a title to the summary in {language_code}. 
    Include an INTRODUCTION, BULLET POINTS if possible, and a CONCLUSION in {language_code}.'''

    response = openai.ChatCompletion.create(
        model=model_name,
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': prompt}
        ],
        temperature=1
    )
    
    return response['choices'][0]['message']['content']

if pdf_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(pdf_file.read())
        pdf_path = tmp_file.name
        loader = PyPDFLoader(pdf_path)
        pages = loader.load_and_split()

if page_selection == "Özet" and pages is not None:
        combined_content = ''.join([p.page_content for p in pages])  # we get entire page data
        texts = text_splitter.split_text(combined_content)
        docs = [Document(page_content=t) for t in texts]
        chain = load_summarize_chain(llm, chain_type="map_reduce")
        summaries = chain.run(docs)
        st.subheader("Özet")
        st.write(summaries)

elif page_selection == "Soru" and pages is not None:
        question = st.text_input("Sorunuzu girin", value="Sorunuzu girin..")
        combined_content = ''.join([p.page_content for p in pages])
        text_splitter = CharacterTextSplitter(
            separator = "\n",
            chunk_size = 1500,
            chunk_overlap  = 200,
            length_function = len,
        )
        texts = text_splitter.split_text(combined_content)
        embeddings = OpenAIEmbeddings(model='gpt-3.5-turbo-instruct', openai_api_key='sk-proj-ANmYN7yLbsp0WeQaY0GUT3BlbkFJQ59X69TRoQKIOPSqs15P')
        document_search = FAISS.from_texts(texts, embeddings)
        chain = load_qa_chain(llm, chain_type="stuff")
        docs = document_search.similarity_search(question)
        summaries = chain.run(input_documents=docs, question=question)
        st.write(summaries)

elif page_selection == "Youtube":
    link = st.text_input('Özetlemek istediğiniz YouTube videosunu girin:')
    if st.button('Başlat'):
        if link:
            try:
                progress = st.progress(0)
                status_text = st.empty()
                status_text.text('yükleniyor...')
                progress.progress(25)
                transcript, language_code = get_transcript(link)
                status_text.text(f'Özetleme...')
                progress.progress(75)
                model_name = 'gpt-3.5-turbo'
                summary = summarize_with_langchain_and_openai(transcript, language_code, model_name)
                status_text.text('Özet:')
                st.markdown(summary)
                progress.progress(100)
            except Exception as e:
                st.write(str(e))

elif page_selection == "Website":
    link = st.text_input('Özetlemek istediğiniz Website linki girin:')
    if st.button('Başlat'):
        if link:
            try:
                loader = WebBaseLoader(link)
                data = loader.load()
                result = type(data),type(data[0])
                text_splitter = CharacterTextSplitter(
                    chunk_size=3000, 
                    chunk_overlap=200,
                    length_function=len,
                )
                texts = text_splitter.split_text(data[0].page_content)
                docs = [Document(page_content=t) for t in texts[:]]
                map_reduce_chain = load_summarize_chain(llm, chain_type="map_reduce")
                output = map_reduce_chain.run(docs)
                st.write(output)

            except Exception as e:
                st.write(str(e))

else:
    time.sleep(30)
    st.warning("PDF dosyası yüklenmedi")


# URL:
# https://blog.langchain.dev/llms-and-sql/
# https://lilianweng.github.io/posts/2023-06-23-agent/

# Video:
# https://www.youtube.com/watch?v=J4RqCSD--Dg&t=1s
