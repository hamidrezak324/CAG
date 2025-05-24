from fastapi import FastAPI, HTTPException
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline
import redis
import json
import pandas as pd
import chromadb
from pydantic import BaseModel

app = FastAPI(title="CAGR API", description="API for calculating CAGR and generating responses")

# مدل برای ورودی API
class CAGRRequest(BaseModel):
    symbol_id: str
    years: int

# اتصال به Redis
try:
    redis_client = redis.Redis(host="host.docker.internal", port=6379, db=0)
    redis_client.ping()
    print("Redis connection successful")
except redis.ConnectionError as e:
    print(f"Redis connection failed: {e}")
    exit(1)

# تابع محاسبه CAGR
def calculate_cagr(symbol_id: str, years: int):
    try:
        cache_key = f"cagr_{symbol_id}_{years}"
        cached_result = redis_client.get(cache_key)
        if cached_result:
            return json.loads(cached_result)

        client = chromadb.PersistentClient(path="./chroma_db")
        collection = client.get_collection("stock_data")
        results = collection.query(
            query_texts=[f"نماد: {symbol_id}"],
            n_results=years * 365,
            where={"symbol_id": symbol_id}
        )

        if not results["documents"]:
            return {"error": "No data found"}

        dates = [meta["date"] for meta in results["metadatas"][0]]
        closes = [float(doc.split("قیمت بسته: ")[1].split(",")[0]) for doc in results["documents"][0]]
        df = pd.DataFrame({"date": dates, "close": closes})

        start_price = df["close"].iloc[0]
        end_price = df["close"].iloc[-1]
        cagr = (end_price / start_price) ** (1 / years) - 1
        result = {"symbol_id": symbol_id, "cagr": cagr, "years": years}
        redis_client.setex(cache_key, 3600, json.dumps(result))
        return result
    except Exception as e:
        return {"error": f"CAGR calculation failed: {e}"}

# تنظیم مدل LangChain
try:
    print("Loading model...")
    model = pipeline("text-generation", model="HooshvareLab/bert-fa-zwnj-base", max_length=100)
    llm = HuggingFacePipeline(pipeline=model)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error setting up model: {e}")
    exit(1)

# تنظیم پرامپت
prompt = PromptTemplate(
    input_variables=["symbol_id", "cagr", "years"],
    template="نرخ رشد سالانه مرکب (CAGR) برای نماد {symbol_id} در {years} سال گذشته {cagr:.2%} بوده است. این یعنی عملکرد {symbol_id} در این دوره {performance} بوده است."
)

# تابع تولید پاسخ
def generate_cag_response(symbol_id: str, years: int):
    try:
        cagr_data = calculate_cagr(symbol_id, years)
        if "error" in cagr_data:
            return cagr_data["error"]

        performance = "خوب" if cagr_data["cagr"] > 0.05 else "ضعیف"
        chain = prompt | llm
        response = chain.invoke({
            "symbol_id": symbol_id,
            "cagr": cagr_data["cagr"],
            "years": years,
            "performance": performance
        })
        return response
    except Exception as e:
        return f"Error generating response: {e}"

# مسیر API
@app.post("/cagr/", response_model=dict)
async def get_cagr(request: CAGRRequest):
    response = generate_cag_response(request.symbol_id, request.years)
    if response.startswith("Error") or "No data found" in response:
        raise HTTPException(status_code=400, detail=response)
    return {"response": response}

# تست محلی
if __name__ == "__main__":
    import uvicorn
    print("Starting test...")
    test_response = generate_cag_response("71806970632897", 5)
    print("Test response:", test_response)
    uvicorn.run(app, host="0.0.0.0", port=8000)