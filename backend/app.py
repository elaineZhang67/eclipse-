from fastapi import FastAPI

from backend.chat_api import router as chat_router
from backend.developer_api import router as developer_router
from backend.user_api import router as user_router


app = FastAPI(
    title="Surveillance Monitoring Backend",
    version="0.1.0",
    description=(
        "Backend service for video processing, persistent timestamped memory, "
        "and user QA over selected videos/timeframes."
    ),
)

app.include_router(developer_router)
app.include_router(user_router)
app.include_router(chat_router)


@app.get("/health")
def health():
    return {"status": "ok"}
