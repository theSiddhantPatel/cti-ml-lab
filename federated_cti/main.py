import sys
import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

# ----- Get client id from command line -----
client_id = int(sys.argv[1])


# ----- Simple neural network -----
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(28 * 28, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        return self.fc(x)


# ----- Load dataset -----
transform = transforms.Compose([transforms.ToTensor()])
dataset = datasets.MNIST(".", train=True, download=True, transform=transform)

# Split dataset into 3 parts
num_clients = 3
data_per_client = len(dataset) // num_clients

start = client_id * data_per_client
end = start + data_per_client

subset = Subset(dataset, range(start, end))
trainloader = DataLoader(subset, batch_size=32, shuffle=True)

model = Net()


def train():
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    model.train()

    for data, target in trainloader:
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()


def get_parameters():
    return [val.cpu().numpy() for val in model.state_dict().values()]


def set_parameters(parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict)


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

    # No accuracy , No loss tracking  Server has no idea if model is improving or not


# ----- Test dataset -----
testset = datasets.MNIST(".", train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=32)


def test():
    model.eval()
    correct, total, loss = 0, 0, 0.0
    loss_fn = nn.CrossEntropyLoss()

    with torch.no_grad():
        for data, target in testloader:
            output = model(data)
            loss += loss_fn(output, target).item()
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)

    accuracy = correct / total
    return loss, accuracy


fl.client.start_numpy_client(server_address="127.0.0.1:8081", client=FlowerClient())
