
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .model_service import get_service
from .schemas import PredictRequest, PredictResponse

app = FastAPI(title="Batangas Dengue Forecast API", version="6.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
service = get_service()

@app.get("/health")
def health():
    return service.health()

@app.get("/locations")
def locations():
    return {"locations": service.available_locations()}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        return service.predict_one(req.municipality, req.barangay, req.horizon_weeks)
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {error}")

@app.get("/summary")
def summary():
    return service.summary()

@app.get("/map-geojson")
def map_geojson():
    return service.map_geojson()

@app.get("/match-report")
def match_report():
    return service.match_report()
