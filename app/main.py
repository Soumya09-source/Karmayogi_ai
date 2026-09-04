from fastapi import FastAPI


app = FastAPI(title="Upskilling Platform API")


@app.get("/health")
def health():
    return {"status": "ok"}