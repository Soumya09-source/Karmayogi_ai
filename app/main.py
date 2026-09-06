from fastapi import FastAPI


from app.routers import auth, assessments, behavioural, recommendations, courses, documents, mcqs, concept_review


app = FastAPI(title="Upskilling Platform API")

app.include_router(auth.router)
app.include_router(assessments.router)
app.include_router(behavioural.router)
app.include_router(recommendations.router)
app.include_router(courses.router)

app.include_router(concept_review.router)
app.include_router(documents.router)
app.include_router(mcqs.router)


@app.get("/health")
def health():
    return {"status": "ok"}