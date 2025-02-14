from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag_processor import RAGProcessor
import uvicorn

app = FastAPI(title="RAG Systems")
rag = RAGProcessor()


# init system
@app.on_event("startup")
async def startup():
    rag.initialize_system("rand_softwareEngineers_id_jd.csv")


# request model
class QueryRequest(BaseModel):
    question: str
    max_results: int = 3


# response model
class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[int]


@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    try:
        # search question
        context = rag.search(request.question, request.max_results)

        # generate response
        answer = rag.generate_response(request.question, context)

        # get source id
        source_ids = list({doc["job_id"] for doc in context})

        # return in form
        return {
            "question": request.question,
            "answer": answer,
            "sources": source_ids
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)