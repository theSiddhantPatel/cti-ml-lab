from collections import OrderedDict
import json
from pathlib import Path

import flwr as fl
import torch

from nsl_kdd import Net


BASE_DIR = Path(__file__).resolve().parent


class CustomStrategy(fl.server.strategy.FedAvg):
    def __init__(self):
        super().__init__()
        self.accuracy_history = []

    def save_metrics(self):
        with (BASE_DIR / "accuracy_history.json").open("w", encoding="utf-8") as file:
            json.dump(self.accuracy_history, file)

    def aggregate_evaluate(self, server_round, results, failures):
        if not results:
            return None, {}

        accuracies = []
        for _, result in results:
            if "accuracy" in result.metrics:
                accuracies.append(result.metrics["accuracy"])

        if accuracies:
            avg_acc = sum(accuracies) / len(accuracies)
            self.accuracy_history.append(avg_acc)
            print(f"\nRound {server_round} Global Accuracy: {avg_acc:.4f}\n")

        return super().aggregate_evaluate(server_round, results, failures)

    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is not None:
            print(f"Saving global model after round {server_round}...")

            ndarrays = fl.common.parameters_to_ndarrays(aggregated_parameters)
            model_state = OrderedDict()

            for key, value in zip(Net().state_dict().keys(), ndarrays):
                model_state[key] = torch.tensor(value)

            torch.save(model_state, BASE_DIR / f"global_model_round_{server_round}.pth")

        return aggregated_parameters, metrics


def main():
    strategy = CustomStrategy()

    fl.server.start_server(
        server_address="127.0.0.1:8081",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )

    strategy.save_metrics()


if __name__ == "__main__":
    main()
