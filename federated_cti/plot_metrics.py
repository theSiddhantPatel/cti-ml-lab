import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
ACCURACY_HISTORY_FILE = BASE_DIR / "accuracy_history.json"
PLOT_FILE = BASE_DIR / "accuracy_plot.png"


def main():
    if not ACCURACY_HISTORY_FILE.exists():
        raise FileNotFoundError(
            f"Could not find metrics file at {ACCURACY_HISTORY_FILE}. "
            "Run the server and clients first to generate accuracy history."
        )

    with ACCURACY_HISTORY_FILE.open("r", encoding="utf-8") as file:
        accuracy = json.load(file)

    rounds = list(range(1, len(accuracy) + 1))

    plt.figure()
    plt.plot(rounds, accuracy, marker="o")
    plt.xlabel("Rounds")
    plt.ylabel("Accuracy")
    plt.title("Federated Learning Accuracy")
    plt.grid()
    plt.tight_layout()
    plt.savefig(PLOT_FILE)
    print(f"Saved accuracy plot to {PLOT_FILE}")


if __name__ == "__main__":
    main()
