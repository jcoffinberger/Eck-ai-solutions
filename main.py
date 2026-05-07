from fastapi import FastAPI, UploadFile, File, HTTPException
from dotenv import load_dotenv
import uvicorn
from rag import get_rag_chain
from agent import get_agent_executor

load_dotenv()

app = FastAPI(
    title="East Coast Kreations - Enterprise Intelligence Agent",
    description="Production RAG + Autonomous Multi-Agent Systems",
    version="1.0"
)

rag_chain = get_rag_chain()
agent_executor = get_agent_executor()

@app.post("/query")
async def query(question: str):
    rag_answer = rag_chain.invoke(question)
    agent_response = agent_executor.invoke({"messages": [{"role": "user", "content": question}]})
    return {
        "rag_answer": rag_answer,
        "agent_answer": agent_response["messages"][-1].content
    }

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    return {"status": "success", "message": f"Processed {file.filename}"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
