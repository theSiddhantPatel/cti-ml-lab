CTI-ML-Lab
Overview

CTI-ML-Lab is a hands-on machine learning project focused on Cyber Threat Intelligence (CTI).
The goal is to learn and apply core ML concepts by detecting security threats from logs and other cyber-security data, starting with classical machine learning models and gradually evolving toward federated learning.

This project is build-first and learning-oriented.

Project Goals

Learn machine learning using real cybersecurity use cases

Detect suspicious or malicious behavior from security data

Build a clean ML pipeline step by step

Add federated learning later for privacy-preserving intelligence sharing

Data Types (Initial Focus)

This project works with simplified and synthetic versions of:

Network logs

Authentication logs (login attempts, failures)

Suspicious URLs

Malware file hashes

Real datasets will be added gradually as the project evolves.

Machine Learning Scope (Phase 1)

Data preprocessing and feature extraction

Classification and anomaly detection

Model evaluation (precision, recall, confusion matrix)

Initial models:

Logistic Regression

Decision Tree

Random Forest

Isolation Forest

Tech Stack

Python

NumPy

Pandas

scikit-learn

Matplotlib

(No deep learning in the initial phase)

Project Structure
cti-ml-lab/
│── data/ # Datasets (CSV, synthetic logs)
│── notebooks/ # Experiments and learning notebooks
│── src/ # Core ML pipeline code
│── models/ # Trained models
│── README.md

Roadmap

Phase 1

Learn ML basics

Build local threat detection models

Evaluate performance on sample data

Phase 2

Improve feature engineering

Add anomaly detection

Create reusable ML pipeline

Phase 3

Introduce federated learning concepts

Simulate distributed clients

Privacy-preserving model updates

Current Status

🚧 In progress
This project is actively being built while learning machine learning fundamentals.

Motivation

Cyber threats are constantly evolving, and sharing intelligence is hard due to privacy concerns.
This project explores how machine learning, and eventually federated learning, can help detect threats while keeping sensitive data local.
