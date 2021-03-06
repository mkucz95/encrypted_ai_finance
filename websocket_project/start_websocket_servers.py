# start remote servers (here hosted on localhost)
# by calling `run_websocket_server.py` for each remote worker
import subprocess
from init import n_workers, base_port

worker_calls = [["python", "run_websocket_server.py",\
                 "--port", f"{base_port+i}", "--id", f"w_{i}"] for i in range(n_workers)]

for w in worker_calls:
    print(f"Starting server for worker: {w[-1]}")
    subprocess.Popen(w)