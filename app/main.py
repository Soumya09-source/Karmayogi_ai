from fastapi import FastAPI

from app.routers import auth, recommendations

app = FastAPI(title="Upskilling Platform API")

app.include_router(auth.router)
app.include_router(recommendations.router)

@app.get("/health")
def health():
    return {"status": "ok"}