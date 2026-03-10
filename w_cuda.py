# w_cuda.py - The GPU Accelerator
import w_operator as base_op  # Your original script
import numpy as np
import time

# Try to import CUDA libraries
try:
    import cupy as cp
    import cupyx.scipy.sparse as csp
    import cupyx.scipy.sparse.linalg as cspla
    HAS_GPU = True
    print("[SYSTEM] GPU Detected. Akatalêptos Warhead Armed.")
except ImportError:
    HAS_GPU = False
    print("[SYSTEM] No GPU found. Running on CPU (Slow Mode).")

def solve_spectrum_gpu(L_scipy_csr, M=32, tol=1e-8):
    """
    Drop-in replacement for the solver using CuPy.
    """
    if HAS_GPU:
        # Move the sparse matrix to GPU memory
        L_gpu = csp.csr_matrix(L_scipy_csr)
        # Solve for smallest magnitude eigenvalues (SM)
        # Note: CuPy eigsh interface is similar to SciPy
        vals_gpu = cspla.eigsh(L_gpu, k=M, which='SA', tol=tol, return_eigenvectors=False)
        # Move result back to CPU numpy array
        return cp.asnumpy(vals_gpu)
    else:
        # Fallback to the original CPU solver
        return base_op.solve_spectrum(L_scipy_csr, M, tol)

def run_job(params):
    """
    Executes a single point sweep.
    """
    k = params['k']
    G1, G2 = params['G1'], params['G2']
    S = params['S']
    lam = params['lambda']
    w_glue = params['w_glue']
    N = 2 # Neighbor depth

    # 1. Build Graph (CPU is fine for structure)
    vertices, edges, b_ids = base_op.build_graph(k, G1, G2, S, N)
    
    # 2. Add Glue (The Topological Twist)
    upd, psi_by = base_op.add_glue_edges(vertices, b_ids, lam, w_glue, G1, G2)
    edges_merged = base_op.merge_edges(edges, upd)
    
    # 3. Build Laplacian
    # We probe the '0-mode' (s=0,0) first as it's most stable
    L, _ = base_op.build_magnetic_laplacian(vertices, edges_merged, s=(0,0), psi_by_id=psi_by)
    
    # 4. Solve Spectrum (The Heavy Lift)
    eigs = solve_spectrum_gpu(L, M=40)
    
    return eigs
