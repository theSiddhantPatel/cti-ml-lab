from pathlib import Path
import subprocess
import sys


NUM_CLIENTS = 3
BASE_DIR = Path(__file__).resolve().parent
MAIN_SCRIPT = BASE_DIR / "main.py"


def main():
    processes = []

    for client_id in range(NUM_CLIENTS):
        print(f"Starting client {client_id}...")
        process = subprocess.Popen(
            [sys.executable, str(MAIN_SCRIPT), str(client_id)],
            cwd=BASE_DIR,
        )
        processes.append(process)

    for process in processes:
        process.wait()


if __name__ == "__main__":
    main()
