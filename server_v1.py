# server.py - The Hive Dispatcher
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import random
import time

app = FastAPI(title="W@Home Dispatcher")

# --- The Job Queue (In-Memory for MVP) ---
# We sweep lambda from 0.4 to 0.6 (the critical zone) in steps of 0.000001
LAMBDA_START = 0.4
LAMBDA_END = 0.6
STEP = 0.000001
current_index = 0

class JobRequest(BaseModel):
    worker_id: str

class Report(BaseModel):
    worker_id: str
    job_id: int
    found_constants: List[str]  # e.g. ["pi", "alpha"]
    eigenvalues: List[float]    # The raw data for verification
    meta: dict

@app.get("/")
def status():
    return {"status": "Hive Online", "jobs_dispatched": current_index}

@app.post("/job")
def get_job(req: JobRequest):
    global current_index
    
    # Generate the next coordinate in the manifold
    if (LAMBDA_START + current_index * STEP) > LAMBDA_END:
        return {"status": "complete"}
    
    target_lambda = LAMBDA_START + current_index * STEP
    job_id = current_index
    current_index += 1
    
    return {
        "job_id": job_id,
        "params": {
            "k": 4,               # Start simple (Level 4)
            "G1": 32,             # Torus Grid size
            "G2": 32,
            "S": 16,              # Time Circle segments
            "lambda": target_lambda,
            "w_glue": 1000.0      # Strong coupling
        }
    }

@app.post("/report")
def submit_report(report: Report):
    # This is where we catch the "Gold"
    if report.found_constants:
        print(f"!!! DISCOVERY !!! Worker {report.worker_id} found {report.found_constants}")
        with open("discoveries.log", "a") as f:
            f.write(f"{time.ctime()} | Job {report.job_id} | {report.found_constants} | Lambda: {report.meta.get('lambda')}\n")
    
    return {"status": "received"}

# Run with: uvicorn server:app --reload
