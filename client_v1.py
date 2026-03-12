# client.py - The Volunteer
import requests
import time
import numpy as np
import w_cuda  # The warhead
import uuid

SERVER_URL = "http://localhost:8081" # Change this to your VPS IP later
WORKER_ID = str(uuid.uuid4())[:8]

# The Holy Grail Targets
CONSTANTS = {
    "phi": 1.6180339887,
    "e": 2.718281828,
    "pi": 3.141592653,
    "alpha_inv": 137.035999,
    "proton_electron": 1836.15267
}
TOLERANCE = 1e-4  # Start loose, tighten later

def check_for_gold(eigs):
    """
    Scans the spectrum ratios for physical constants.
    """
    eigs = np.sort(eigs[eigs > 1e-9]) # Remove zero modes
    found = []
    
    # Check simple ratios lambda_j / lambda_i
    for i in range(len(eigs)):
        for j in range(i+1, len(eigs)):
            ratio = eigs[j] / eigs[i]
            
            for name, val in CONSTANTS.items():
                if abs(ratio - val) < TOLERANCE:
                    found.append(f"{name} (ratio={ratio:.5f})")
                
                # Also check accumulated ratios (cumulative mass)
                # Sometimes the constant is the sum of eigenvalues
    
    return found

def main():
    print(f"[*] Worker {WORKER_ID} connecting to Hive at {SERVER_URL}")
    
    while True:
        try:
            # 1. Get Job
            resp = requests.post(f"{SERVER_URL}/job", json={"worker_id": WORKER_ID})
            if resp.status_code != 200:
                print("Error connecting to hive.")
                time.sleep(5)
                continue
                
            data = resp.json()
            if data.get("status") == "complete":
                print("All jobs complete. Sleeping.")
                time.sleep(60)
                continue
                
            job_id = data["job_id"]
            params = data["params"]
            
            print(f"[-] Processing Job {job_id} | Lambda: {params['lambda']:.6f} ...")
            
            # 2. Run Warhead
            start_time = time.time()
            eigs = w_cuda.run_job(params)
            duration = time.time() - start_time
            
            # 3. Check for Resonance
            hits = check_for_gold(eigs)
            
            # 4. Visual Feedback (The "Virus")
            if hits:
                print(f"\n[!!!] RESONANCE DETECTED IN JOB {job_id}: {hits}\n")
            else:
                print(f"    -> Done in {duration:.2f}s. No constants found.")
                
            # 5. Report
            requests.post(f"{SERVER_URL}/report", json={
                "worker_id": WORKER_ID,
                "job_id": job_id,
                "found_constants": hits,
                "eigenvalues": eigs.tolist(),
                "meta": params
            })
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"[!] Error: {e}")
            time.sleep(5)

if __name__ == "__main__":
    main()
