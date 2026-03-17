import sys
import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ----- Get client id -----
if len(sys.argv) < 2:
    print("Usage: python main.py <client_id>")
    sys.exit()

client_id = int(sys.argv[1])


# ----- Model -----
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(41, 10)

    def forward(self, x):
        return self.fc(x)


model = Net()


# ----- Load NSL-KDD -----
def load_nsl_kdd(train=True):
    file = "data/KDDTrain+.txt" if train else "data/KDDTest+.txt"

    columns = [f"f{i}" for i in range(41)] + ["label", "difficulty"]
    df = pd.read_csv(file, names=columns)

    df.drop("difficulty", axis=1, inplace=True)

    # Encode categorical features
    for col in df.select_dtypes(include=["object"]).columns:
        if col != "label":
            df[col] = LabelEncoder().fit_transform(df[col])

    # Encode labels
    df["label"] = LabelEncoder().fit_transform(df["label"])

    X = df.drop("label", axis=1).values
    y = df["label"].values

    # Normalize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    return X, y


# ----- Prepare client data -----
X, y = load_nsl_kdd(train=True)

num_clients = 3
data_per_client = len(X) // num_clients

start = client_id * data_per_client
end = start + data_per_client

X_client = X[start:end]
y_client = y[start:end]

trainloader = DataLoader(list(zip(X_client, y_client)), batch_size=32, shuffle=True)


# ----- Test dataset -----
X_test, y_test = load_nsl_kdd(train=False)
testloader = DataLoader(list(zip(X_test, y_test)), batch_size=32)


# ----- Training -----
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


# ----- Testing -----
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


# ----- FL helpers -----
def get_parameters():
    return [val.cpu().numpy() for val in model.state_dict().values()]


def set_parameters(parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict)


# ----- Flower Client -----
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


# ----- Start Client -----
fl.client.start_numpy_client(server_address="127.0.0.1:8081", client=FlowerClient())
