from fastapi import FastAPI

from app.routers import auth
from app.routers import recommendations
from app.routers import courses

app = FastAPI(title="Upskilling Platform API")

app.include_router(auth.router)
app.include_router(recommendations.router)
app.include_router(courses.router)


@app.get("/health")
def health():
    return {"status": "ok"}