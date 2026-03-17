import flwr as fl
import torch
import torch.nn as nn
from collections import OrderedDict
import json


# Model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(28 * 28, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        return self.fc(x)


# Custom strategy
class CustomStrategy(fl.server.strategy.FedAvg):

    def __init__(self):
        super().__init__()
        self.accuracy_history = []  # store global accuracy

    def save_metrics(self):
        with open("accuracy_history.json", "w") as f:
            json.dump(self.accuracy_history, f)

    def aggregate_evaluate(self, server_round, results, failures):
        if not results:
            return None, {}

        accuracies = [res.metrics["accuracy"] for _, res in results]
        avg_accuracy = sum(accuracies) / len(accuracies)
        self.accuracy_history.append(avg_accuracy)  # store it

        print(f"\n Round {server_round} Global Accuracy: {avg_accuracy:.4f}\n")

        return super().aggregate_evaluate(server_round, results, failures)

    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is not None:
            print(f"Saving global model after round {server_round}...")

            ndarrays = fl.common.parameters_to_ndarrays(aggregated_parameters)

            model_dict = OrderedDict()
            for k, v in zip(Net().state_dict().keys(), ndarrays):
                model_dict[k] = torch.tensor(v)

            torch.save(model_dict, f"global_model_round_{server_round}.pth")

        return aggregated_parameters, metrics


strategy = CustomStrategy()

fl.server.start_server(
    server_address="127.0.0.1:8081",
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=strategy,
)

# after training, save the accuracy history
strategy.save_metrics()
