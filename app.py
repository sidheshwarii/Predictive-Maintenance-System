"""
app.py
------
FastAPI backend: authentication (signup/login/forgot/reset password) plus
a schema-driven /predict endpoint for Remaining Useful Life (RUL)
regression on the NASA CMAPSS turbofan dataset.

Run with:
    uvicorn app:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import joblib
import numpy as np
import json
import os

import auth

MODEL_DIR = "model"

app = FastAPI(title="Turbofan RUL Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

rf_model = None
lr_model = None
scaler = None
schema: Dict[str, Any] = {}


@app.on_event("startup")
def load_artifacts():
    global rf_model, lr_model, scaler, schema
    try:
        rf_model = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))
        lr_model = joblib.load(os.path.join(MODEL_DIR, "linear_model.pkl"))
        scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
        with open(os.path.join(MODEL_DIR, "schema.json")) as f:
            schema = json.load(f)
        print("Model artifacts + schema loaded (random_forest + linear_regression).")
    except FileNotFoundError:
        print("WARNING: model artifacts not found. Run train_model.py first.")


# ---------- Auth (unchanged in shape from the factory version) ----------

class SignupRequest(BaseModel):
    username: str
    password: str
    email: str


class LoginRequest(BaseModel):
    username: str
    password: str


class ForgotPasswordRequest(BaseModel):
    username_or_email: str


class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str


@app.post("/signup")
def signup(req: SignupRequest):
    try:
        auth.create_user(req.username, req.password, req.email)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    token = auth.create_session(req.username)
    return {"token": token, "username": req.username}


@app.post("/login")
def login(req: LoginRequest):
    if not auth.verify_user(req.username, req.password):
        raise HTTPException(status_code=401, detail="Invalid username or password.")
    token = auth.create_session(req.username)
    return {"token": token, "username": req.username}


@app.post("/logout")
def logout_route(authorization: Optional[str] = Header(None)):
    token = (authorization or "").replace("Bearer ", "")
    auth.logout(token)
    return {"message": "Logged out."}


@app.get("/me")
def me(authorization: Optional[str] = Header(None)):
    token = (authorization or "").replace("Bearer ", "")
    username = auth.get_username_from_token(token)
    if not username:
        raise HTTPException(status_code=401, detail="Not authenticated.")
    return {"username": username}


@app.post("/forgot-password")
def forgot_password(req: ForgotPasswordRequest):
    token = auth.create_reset_token(req.username_or_email)
    if not token:
        return {"message": "If that account exists, a reset link was sent."}
    return {
        "message": "Reset link generated (would normally be emailed).",
        "reset_token": token,
    }


@app.post("/reset-password")
def reset_password_route(req: ResetPasswordRequest):
    ok = auth.reset_password(req.token, req.new_password)
    if not ok:
        raise HTTPException(status_code=400, detail="Reset link is invalid or expired.")
    return {"message": "Password reset. You can now log in."}


# ---------- Prediction (RUL regression) ----------

def risk_tier(rul: float):
    if rul > 50:
        return "Normal"
    if rul > 20:
        return "Maintenance Soon"
    return "High Risk"


@app.get("/")
def root():
    return {"message": "Turbofan RUL Prediction API is running."}


@app.get("/schema")
def get_schema():
    if not schema:
        raise HTTPException(status_code=503, detail="Model not trained yet. Run train_model.py.")
    return schema


@app.post("/predict")
def predict(payload: Dict[str, Any]):
    """
    payload: { "reading": {feature: value, ...}, "model_type": "random_forest" | "linear_regression" }
    model_type defaults to "random_forest" if omitted.
    """
    if rf_model is None or lr_model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run train_model.py first.")

    model_type = payload.get("model_type", "random_forest")
    reading = payload.get("reading", payload)  # tolerate flat payloads too

    numeric_features = schema["numeric_features"]
    row = []
    for feature in numeric_features:
        value = reading.get(feature)
        if value is None:
            raise HTTPException(status_code=400, detail=f"Missing field: {feature}")
        row.append(float(value))

    features_scaled = scaler.transform(np.array([row]))

    if model_type == "linear_regression":
        chosen_model = lr_model
    elif model_type == "random_forest":
        chosen_model = rf_model
    else:
        raise HTTPException(status_code=400, detail="model_type must be 'random_forest' or 'linear_regression'.")

    predicted_rul = max(0.0, float(chosen_model.predict(features_scaled)[0]))

    return {
        "predicted_rul": round(predicted_rul, 1),
        "status": risk_tier(predicted_rul),
        "model_used": model_type,
    }
