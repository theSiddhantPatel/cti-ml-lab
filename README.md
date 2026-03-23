# Federated Learning Intrusion Detection with Flower and PyTorch

## Project Overview

This project builds a simple federated learning system using **Flower** and **PyTorch** for **network intrusion detection** on the **NSL-KDD** dataset. Multiple clients train the same model on their own local data partitions, while a central server collects and aggregates the model updates to create a better global model.

The main idea is to train a shared model **without sending raw client data to the server**. This makes the project a good beginner-friendly example of privacy-aware distributed machine learning for cybersecurity use cases.

## What Is Federated Learning?

Federated learning is a machine learning approach where many clients train a model locally on their own data.  
Instead of sharing datasets, each client only sends model updates to a server, and the server combines them to improve the global model.

## Features

- Multi-client federated learning setup
- Custom Flower strategy for server-side aggregation
- Global model aggregation across training rounds
- Accuracy tracking after each round
- Accuracy visualization using `matplotlib`
- Global model saving after every round
- NSL-KDD based intrusion detection workflow

## Tech Stack

- Python
- Flower
- PyTorch
- Pandas
- Scikit-learn
- NumPy
- Matplotlib

## Project Structure

```text
cti-ml-lab/
├── federated_cti/
│   ├── main.py           # Client code for local training and evaluation
│   ├── server.py         # Server setup, custom strategy, and aggregation logic
│   ├── run_clients.py    # Starts multiple clients automatically
│   └── plot_metrics.py   # Plots accuracy across training rounds
├── requirements.txt      # Project dependencies
└── README.md             # Project documentation
```

## How the Project Works

1. The server starts and waits for clients to connect.
2. Each client loads its own part of the NSL-KDD training data.
3. Clients train the model locally using PyTorch.
4. The server collects the updated model parameters from all clients.
5. The server aggregates these updates into one global model.
6. This process repeats for multiple rounds.
7. After each round, the global model is saved and accuracy is tracked.

## Installation

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd cti-ml-lab
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the virtual environment

On Windows:

```bash
venv\Scripts\activate
```

On macOS/Linux:

```bash
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

## How to Run the Project

Open a terminal in the repository root:

```bash
cd cti-ml-lab
```

### Step 1: Start the server

```bash
python federated_cti/server.py
```

The server will start on:

```text
127.0.0.1:8081
```

### Step 2: Start the clients

Open another terminal in the same folder and run:

```bash
python federated_cti/run_clients.py
```

This script starts 3 clients. Each client trains on a different portion of the NSL-KDD dataset.

### Step 3: Plot the accuracy graph

After training is complete, run:

```bash
python federated_cti/plot_metrics.py
```

This will generate an accuracy plot from the saved training history.

## Example Output

During training, you may see output similar to this:

```text
Round 1 Global Accuracy: 0.7821
Round 2 Global Accuracy: 0.8467
Round 3 Global Accuracy: 0.8794
```

This shows that the global model improves over time as the server combines learning from all clients.

## Files Generated After Training

- `accuracy_history.json` - Stores accuracy values for each round
- `global_model_round_1.pth` - Saved global model after round 1
- `global_model_round_2.pth` - Saved global model after round 2
- `global_model_round_3.pth` - Saved global model after round 3
- `accuracy_plot.png` - Accuracy graph created by `plot_metrics.py`

## Future Improvements

- Add support for more clients and flexible client selection
- Use a deeper neural network for better performance
- Add support for non-IID data distribution
- Track loss along with accuracy
- Save training logs in a more detailed format
- Add configuration files for easy experiment setup
- Support GPU training for faster local updates

## Why This Project Is Useful

This project is a simple starting point for learning how federated learning works in practice. It helps beginners understand how local training, server aggregation, and round-based learning come together in a real implementation.

## Conclusion

This federated learning system demonstrates how Flower and PyTorch can be used together to train a shared model while keeping client data local. It is small, practical, and easy to extend for more advanced federated learning experiments.
