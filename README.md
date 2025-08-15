# Privacy-Enhanced-Multi-Institutional-Financial-Fraud-Prediction-Model
This project implements a Hierarchical Federated Learning (HFL) framework to perform loan default prediction while preserving data privacy across multiple institutions. Unlike traditional centralized models, HFL enables collaborative model training without raw data sharing, making it suitable for privacy-sensitive domains such as finance.

Key Features:

Hierarchical Federated Learning (HFL): Multi-level aggregation with clients, middle servers, and a global server for scalability.

Privacy-Preserving Training: Simulates Non-IID and imbalanced datasets across 12 clients to mimic real-world multi-institution scenarios.

Machine Learning Model: A custom PyTorch neural network with batch normalization, dropout, and adaptive optimization for binary loan default classification.

Federated Averaging (FedAvg): Implements FedAvg with dynamic client sampling, variable local epochs, and uneven quantity skew distribution.

Performance Tracking: Evaluates Accuracy, Loss, F1-Score, and AUC per communication round and analyzes client drift.

Visualization: Generates plots for model performance trends and final client model distributions.

Technical Stack:

Python, PyTorch, Scikit-learn, Pandas, NumPy, Matplotlib

Federated Learning concepts: FedAvg, Non-IID simulation, hierarchical aggregation

Privacy-preserving distributed ML architecture
