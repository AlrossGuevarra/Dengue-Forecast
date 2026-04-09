from pydantic import BaseModel, Field

class PredictRequest(BaseModel):
    municipality: str = Field(..., min_length=1)
    barangay: str = Field(..., min_length=1)
    horizon_weeks: int = Field(default=2, ge=2, le=2)

class SupportingIndicators(BaseModel):
    rainfall_mm: float | None = None
    temperature_c: float | None = None
    humidity_percent: float | None = None
    population: int | None = None
    cases_per_1000: float | None = None

class PredictResponse(BaseModel):
    municipality: str
    barangay: str
    forecast_year: int
    forecast_week: int
    context_year: int
    context_week: int
    risk_label: str
    prediction_score: float
    probabilities: dict
    supporting_indicators: SupportingIndicators
    trend: str | None = None
    last_observed_cases: int | None = None
    weekly_trend: list[int] = []
    historical_total_cases: int | None = None
    historical_2025_cases: int | None = None
    reason_summary: str
    notes: str
