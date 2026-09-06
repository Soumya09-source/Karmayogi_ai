from fastapi import FastAPI

from app.routers import auth, assessments, behavioural

app = FastAPI(title="Upskilling Platform API")

app.include_router(auth.router)
app.include_router(assessments.router)
app.include_router(behavioural.router)

@app.get("/health")
def health():
    return {"status": "ok"}