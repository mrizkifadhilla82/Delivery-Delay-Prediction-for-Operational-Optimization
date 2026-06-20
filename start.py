import subprocess
import sys
import time
import os

def start_services():
    print("Starting Delivery Delay Prediction MLOps Services...")
    
    # Path to virtual env python/scripts
    if os.name == 'nt':
        python_exe = os.path.join("venv", "Scripts", "python.exe")
        uvicorn_exe = os.path.join("venv", "Scripts", "uvicorn.exe")
        streamlit_exe = os.path.join("venv", "Scripts", "streamlit.exe")
    else:
        python_exe = os.path.join("venv", "bin", "python")
        uvicorn_exe = os.path.join("venv", "bin", "uvicorn")
        streamlit_exe = os.path.join("venv", "bin", "streamlit")

    # Verify virtualenv executables exist, otherwise fallback to system path
    if not os.path.exists(python_exe):
        python_exe = sys.executable
        uvicorn_exe = "uvicorn"
        streamlit_exe = "streamlit"

    processes = []
    try:
        # 1. Start FastAPI API Server
        print("Launching FastAPI Backend on http://127.0.0.1:8000 ...")
        api_proc = subprocess.Popen([
            uvicorn_exe, "app.api:app", 
            "--host", "127.0.0.1", 
            "--port", "8000",
            "--log-level", "info"
        ])
        processes.append(api_proc)
        
        # Give API a moment to spin up and load the TF model
        time.sleep(3)
        
        # 2. Start Streamlit UI Dashboard
        print("Launching Streamlit UI on http://localhost:8501 ...")
        streamlit_proc = subprocess.Popen([
            streamlit_exe, "run", "app/app.py"
        ])
        processes.append(streamlit_proc)
        
        print("\nPress Ctrl+C to terminate both services.\n")
        
        # Keep main thread alive and monitor processes
        while True:
            for p in processes:
                if p.poll() is not None:
                    print(f"\nProcess {p.pid} terminated unexpectedly. Shutting down...")
                    raise KeyboardInterrupt
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nShutting down services...")
        for p in processes:
            try:
                p.terminate()
                p.wait(timeout=2)
            except Exception:
                p.kill()
        print("Services shut down successfully.")

if __name__ == "__main__":
    start_services()
