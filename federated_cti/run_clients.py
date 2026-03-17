import subprocess
import sys

num_clients = 3
processes = []

for i in range(num_clients):
    print(f"Starting client {i}...")
    p = subprocess.Popen([sys.executable, "main.py", str(i)])
    processes.append(p)

for p in processes:
    p.wait()
