# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from .database import connect_to_mongodb, close_mongodb_connection
# from .routes import auth, stock
# # from app.stock.routes import router as stock_router

# app = FastAPI()

# # Configure CORS
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
#     expose_headers=["*"]
# )

# @app.on_event("startup")
# async def startup_db_client():
#     await connect_to_mongodb()

# @app.on_event("shutdown")
# async def shutdown_db_client():
#     await close_mongodb_connection()


# app.include_router(auth.router, prefix="/auth", tags=["authentication"])
# app.include_router(stock.router, prefix="/stocks", tags=["stocks"])
# # # app.include_router(stock_router, tags=["stocks"])
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.database import connect_to_mongodb, close_mongodb_connection
from app.routes.auth import router as auth_router
from app.routes.stock import router as stock_router
from app.routes.chart import router as chart_router

app = FastAPI(title="NSE Stock Price Tracker")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this with your React app URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_db_client():
    await connect_to_mongodb()

@app.on_event("shutdown")
async def shutdown_db_client():
    await close_mongodb_connection()

# Include routers
app.include_router(auth_router, prefix="/auth", tags=["authentication"])
app.include_router(stock_router, prefix="/stocks", tags=["stocks"])
app.include_router(chart_router, prefix="/chart", tags=["chart"])
