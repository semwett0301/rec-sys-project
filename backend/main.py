# main.py

from fastapi import FastAPI, Depends
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

from recommendation_service import RecommendationService

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


recommendation_service = RecommendationService()


def get_recommendation_service() -> RecommendationService:
    return recommendation_service


class RecommendationResponse(BaseModel):
    recommendations: list[str]


@app.get("/recommend/{model}/{dataset}", response_model=RecommendationResponse)
def greet(model: str, dataset: str, user_id: str, top_n: int = 10,
          service: RecommendationService = Depends(get_recommendation_service)):
    result = service.recommend(model, dataset, user_id, top_n)
    print("RECOMMEND RESULT:", result, type(result))

    return JSONResponse(content={"recommendations": result},
                        media_type="application/json")
