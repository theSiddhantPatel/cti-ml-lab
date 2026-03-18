import sys
import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from nsl_kdd import NUM_CLIENTS, Net, prepare_datasets, split_train_dataset

# ---- Device ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Client ID ----
if len(sys.argv) < 2:
    print("Usage: python main.py <client_id>")
    sys.exit(1)

client_id = int(sys.argv[1])

# ---- Load Data ----
model = Net().to(device)

train_dataset, test_dataset, _, label_encoder, _ = prepare_datasets()
client_train_dataset = split_train_dataset(train_dataset, client_id, NUM_CLIENTS)

trainloader = DataLoader(client_train_dataset, batch_size=32, shuffle=True)
testloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(
    f"Client {client_id}: loaded {len(client_train_dataset)} training rows, "
    f"{len(test_dataset)} test rows, {len(label_encoder.classes_)} classes"
)


# ---- Train ----
def train(epochs=3):
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for data, target in trainloader:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Client {client_id} Epoch {epoch+1} Loss: {total_loss:.4f}")


# ---- Test ----
def test():
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    loss_fn = nn.CrossEntropyLoss()

    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            total_loss += loss_fn(output, target).item()
            predictions = output.argmax(dim=1)

            correct += (predictions == target).sum().item()
            total += target.size(0)

    return total_loss / len(testloader), correct / total


# ---- Flower helpers ----
def get_parameters():
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)


# ---- Client ----
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return get_parameters()

    def fit(self, parameters, config):
        print(f"Client {client_id}: FIT STARTED")
        set_parameters(parameters)
        train()
        print(f"Client {client_id}: FIT FINISHED")
        return get_parameters(), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        set_parameters(parameters)
        loss, accuracy = test()
        print(f"Client {client_id}: Accuracy = {accuracy:.4f}")
        return loss, len(testloader.dataset), {"accuracy": accuracy}


# ---- Start ----
if __name__ == "__main__":
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8081",
        client=FlowerClient(),
    )
