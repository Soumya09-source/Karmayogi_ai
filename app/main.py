from fastapi import FastAPI

from app.routers import auth

app = FastAPI(title="Upskilling Platform API")

app.include_router(auth.router)


@app.get("/health")
def health():
    return {"status": "ok"}