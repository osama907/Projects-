import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
import os
import re

# Load .env variables (e.g., OpenAI API key)
load_dotenv()

st.set_page_config(page_title="YouTube Video QA", layout="wide")
st.title("YouTube Video Q&A using LangChain")

def extract_video_id(url_or_id):
    match = re.search(r"(?:v=|be/)([\w-]{11})", url_or_id)
    if match:
        return match.group(1)
    return url_or_id.strip()

video_url = st.text_input("Enter YouTube Video URL or ID (e.g., `Gfr50f6ZBvo` or full URL)", "")

if video_url:
    video_id = extract_video_id(video_url)

    if len(video_id) != 11:
        st.error("Invalid video ID. Please enter a valid YouTube video ID or URL.")
    else:
        try:
            st.info("Fetching transcript...")
            transcript = YouTubeTranscriptApi().fetch(video_id)
            full_text = " ".join([t.text for t in transcript])
            st.success("Transcript fetched and processed!")

            # Chunking
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunk = splitter.create_documents([full_text])

            # Embeddings and Vector store
            embeddings = OpenAIEmbeddings()
            vector_store = FAISS.from_documents(chunk, embeddings)
            retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={'k': 3})

            # LLM & Prompt
            llm = ChatOpenAI()
            parser = StrOutputParser()
            prompt = PromptTemplate(
                template="""
You are a helpful assistant.
Answer ONLY from the provided transcript context.
If the context is insufficient, just say you don't know.

{context}
Question: {question}
""",
                input_variables=['context', 'question']
            )

            def retreived_doc(retrieved_docs):
                return "\n\n".join(doc.page_content for doc in retrieved_docs)

            parallel_chain = RunnableParallel({
                'context': retriever | RunnableLambda(retreived_doc),
                'question': RunnablePassthrough()
            })

            main_chain = parallel_chain | prompt | llm | parser

            question = st.text_input("Ask a question based on the video transcript:")

            if question:
                st.info("Generating answer...")
                result = main_chain.invoke(question)
                st.markdown("### Answer")
                st.write(result)

        except TranscriptsDisabled:
            st.error("This video has no captions/transcript available.")
        except Exception as e:
            st.error(f"Error: {str(e)}")
