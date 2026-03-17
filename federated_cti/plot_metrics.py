import json
import matplotlib.pyplot as plt

# Load accuracy
with open("accuracy_history.json", "r") as f:
    accuracy = json.load(f)

rounds = list(range(1, len(accuracy) + 1))

plt.figure()
plt.plot(rounds, accuracy, marker="o")
plt.xlabel("Rounds")
plt.ylabel("Accuracy")
plt.title("Federated Learning Accuracy")
plt.grid()

plt.savefig("accuracy_plot.png")
plt.show()
