from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import TensorDataset


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
TRAIN_FILE = DATA_DIR / "KDDTrain+.txt"
TEST_FILE = DATA_DIR / "KDDTest+.txt"

INPUT_DIM = 41
CLASS_NAMES = ["normal", "dos", "probe", "r2l", "u2r"]
NUM_CLASSES = len(CLASS_NAMES)
NUM_CLIENTS = 3

ATTACK_GROUPS = {
    "normal": {"normal"},
    "dos": {
        "back",
        "land",
        "neptune",
        "pod",
        "smurf",
        "teardrop",
        "mailbomb",
        "apache2",
        "processtable",
        "udpstorm",
    },
    "probe": {"satan", "ipsweep", "nmap", "portsweep", "mscan", "saint"},
    "r2l": {
        "guess_passwd",
        "ftp_write",
        "imap",
        "phf",
        "multihop",
        "warezmaster",
        "warezclient",
        "spy",
        "xlock",
        "xsnoop",
        "snmpguess",
        "snmpgetattack",
        "httptunnel",
        "sendmail",
        "named",
        "worm",
    },
    "u2r": {
        "buffer_overflow",
        "loadmodule",
        "rootkit",
        "perl",
        "sqlattack",
        "xterm",
        "ps",
    },
}

ATTACK_TO_CATEGORY = {
    attack_name: category
    for category, attack_names in ATTACK_GROUPS.items()
    for attack_name in attack_names
}


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            #  nn.Linear(INPUT_DIM, NUM_CLASSES)  # simple linear model
            nn.Linear(INPUT_DIM, 64),  # deep MLP total 3 layers
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, NUM_CLASSES),
        )

    def forward(self, x):
        return self.network(x)


def _load_dataframe(path: Path) -> pd.DataFrame:
    columns = [f"f{i}" for i in range(INPUT_DIM)] + ["label", "difficulty"]
    dataframe = pd.read_csv(path, names=columns)
    dataframe.drop(columns=["difficulty"], inplace=True)
    original_labels = dataframe["label"].copy()
    dataframe["label"] = dataframe["label"].map(ATTACK_TO_CATEGORY)

    if dataframe["label"].isna().any():
        missing = sorted(original_labels[dataframe["label"].isna()].unique())
        raise ValueError(f"Unmapped NSL-KDD labels found in {path.name}: {missing}")

    return dataframe


def prepare_datasets():
    train_df = _load_dataframe(TRAIN_FILE)
    test_df = _load_dataframe(TEST_FILE)

    categorical_columns = (
        train_df.drop(columns=["label"])
        .select_dtypes(include=["object", "string"])
        .columns
    )

    feature_encoders = {}
    for column in categorical_columns:
        encoder = LabelEncoder()
        encoder.fit(pd.concat([train_df[column], test_df[column]], axis=0))
        train_df[column] = encoder.transform(train_df[column])
        test_df[column] = encoder.transform(test_df[column])
        feature_encoders[column] = encoder

    label_encoder = LabelEncoder()
    label_encoder.fit(CLASS_NAMES)
    train_df["label"] = label_encoder.transform(train_df["label"])
    test_df["label"] = label_encoder.transform(test_df["label"])

    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_df.drop(columns=["label"]).values)
    x_test = scaler.transform(test_df.drop(columns=["label"]).values)

    y_train = train_df["label"].values
    y_test = test_df["label"].values

    train_dataset = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    test_dataset = TensorDataset(
        torch.tensor(x_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long),
    )

    return train_dataset, test_dataset, feature_encoders, label_encoder, scaler


def split_train_dataset(train_dataset, client_id, num_clients=NUM_CLIENTS):
    if client_id < 0 or client_id >= num_clients:
        raise ValueError(f"client_id must be between 0 and {num_clients - 1}")

    # ---- Shuffle dataset ----
    features, labels = train_dataset.tensors
    perm = torch.randperm(len(features))

    features = features[perm]
    labels = labels[perm]

    total_samples = len(features)
    start = client_id * total_samples // num_clients
    end = (client_id + 1) * total_samples // num_clients

    return TensorDataset(features[start:end], labels[start:end])
