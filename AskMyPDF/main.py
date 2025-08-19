from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from pydantic import BaseModel
import os
import markdown
import shutil

load_dotenv()

app = FastAPI()

# CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all during development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global vars
vector_store = None
qa_chain = None
UPLOAD_DIR = "uploads"

# Ensure upload directory exists
os.makedirs(UPLOAD_DIR, exist_ok=True)


def process_pdf(file_path: str):
    global vector_store, qa_chain

    loader = PyPDFLoader(file_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)

    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    llm = ChatOpenAI()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)


@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    process_pdf(file_path)

    return {"message": "PDF uploaded and processed successfully."}


class QuestionRequest(BaseModel):
    question: str


@app.post("/ask/")
async def ask_question(payload: QuestionRequest):
    global qa_chain

    if qa_chain is None:
        return JSONResponse(status_code=400, content={"error": "PDF not uploaded yet."})

    try:
        answer = qa_chain.invoke(payload.question)
        return {
            "query": payload.question,
            "answer": answer
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/summarize/", response_class=HTMLResponse)
def summarize():
    global vector_store
    try:
        if vector_store is None:
            raise HTTPException(status_code=400, detail="No PDF uploaded yet.")

        # Search for most relevant chunks
        docs = vector_store.similarity_search("what is this document about")
        content = "\n".join([doc.page_content for doc in docs])

        llm = ChatOpenAI()

        # Step 1: Identify document type
        doc_type_prompt = PromptTemplate.from_template("""
        You're an AI expert in reading PDFs. Based on the content below, tell me what type of document this is.
        Choose one: "resume", "research paper", "book", or "unknown".
        Text:
        {text}
        """)
        detect_prompt = doc_type_prompt.format(text=content)
        type_result = llm.invoke(detect_prompt)

        doc_type = type_result.content.lower() if hasattr(type_result, "content") else str(type_result).lower()

        # Step 2: Generate summary based on type
        if "resume" in doc_type:
            universal_prompt = PromptTemplate.from_template("""
               You are a resume extraction assistant. Given the following resume text, extract the information and return it in this exact Markdown format using double asterisks (**) to bold both field labels and values.
               **Name:** [Full Name] 
               **Email:** [Email if present]  
               **Phone:** [Phone Number if present]  
               **Education:**  
               - [Degree, Institution, Year if available]
               **Skills:**  
               - [List relevant skills]


             Resume:
             {text}
            """)
        else:
            universal_prompt = PromptTemplate.from_template(
                """ 
                You are an expert at extracting information from messy PDF text. The text below is jumbled from PDF extraction.

                SCAN the entire text and find these elements wherever they appear:

                1. TITLE: Look for the main paper title (usually the longest descriptive phrase)
                2. AUTHORS: Look for people's names (usually after "Authors" or near the title)  
                3. ABSTRACT/SUMMARY: Look for the paragraph that describes what the paper is about
                4. OBJECTIVE/GOAL: Look for sentences mentioning "goal", "objective", "aim", "purpose"
                5. CONCLUSION: Look for final results or conclusions
                6. KEYWORDS: Look for technical terms or a keywords section

                Output format (put each value on SAME line as the bold label):

                **Title :** [extracted title]

                **Author :** [extracted authors]

                **Summary :** [extracted summary/abstract]

                **Objective :** [extracted objective]

                **Conclusion :** [extracted conclusion]

                **Keywords :** [extracted keywords]

                MESSY PDF TEXT:
                {text}

                Remember: IGNORE the formatting mess and focus on finding the actual content!
               """)

        final_prompt = universal_prompt.format(text=content)
        summary = llm.invoke(final_prompt)
        summary_text = summary.content if hasattr(summary, "content") else str(summary)
        html = markdown.markdown(summary_text)
        return HTMLResponse(content=html)



    except Exception as e:
        print("Summarize error:", e)
        raise HTTPException(status_code=500, detail="Summarization failed")


@app.get("/clear-vectorstore/")
async def clear_store():
    global vector_store, qa_chain

    vector_store = None
    qa_chain = None

    if os.path.exists(UPLOAD_DIR):
        for filename in os.listdir(UPLOAD_DIR):
            file_path = os.path.join(UPLOAD_DIR, filename)
            os.remove(file_path)

    return {"message": "Vector store, uploaded PDFs, and session cleared."}


@app.get("/")
def root():
    return {"message": "API is running. Use /upload-pdf/, /ask/, /summarize/, or /clear-vectorstore/"}