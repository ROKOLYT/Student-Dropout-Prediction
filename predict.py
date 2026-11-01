import pickle
from typing import Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import xgboost as xgb

# Initialize FastAPI application
app = FastAPI(
    title="Student Dropout Prediction API",
    description="Early Warning API to predict student dropout risk in higher education",
    version="1.0.0"
)

MODEL_FILE = "model.bin"

# Load model and DictVectorizer
try:
    with open(MODEL_FILE, "rb") as f_in:
        dv, model = pickle.load(f_in)
    feature_names = dv.get_feature_names_out().tolist()
    print(f"Loaded model and vectorizer from '{MODEL_FILE}'. Features count: {len(feature_names)}")
except Exception as e:
    raise RuntimeError(f"Failed to load model file '{MODEL_FILE}': {e}")


class StudentProfile(BaseModel):
    # Allow arbitrary student feature dictionaries
    model_config = {"extra": "allow"}


class PredictionResponse(BaseModel):
    dropout_probability: float
    dropout_prediction: bool
    risk_level: str


@app.get("/")
def health_check():
    return {
        "status": "healthy",
        "service": "Student Dropout Prediction API",
        "version": "1.0.0"
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(student: Dict[str, Any]):
    if not student:
        raise HTTPException(status_code=400, detail="Student payload cannot be empty.")
    
    try:
        # Preprocess input with DictVectorizer
        X = dv.transform([student])
        dmatrix = xgb.DMatrix(X, feature_names=feature_names)
        
        # Predict probability
        prob = float(model.predict(dmatrix)[0])
        is_dropout = prob >= 0.5
        
        return PredictionResponse(
            dropout_probability=round(prob, 4),
            dropout_prediction=is_dropout,
            risk_level="High Risk" if is_dropout else "Low Risk"
        )
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(err)}")


if __name__ == "__main__":
    import uvicorn
    port = int(8000)
    uvicorn.run("predict:app", host="0.0.0.0", port=port, reload=False)
