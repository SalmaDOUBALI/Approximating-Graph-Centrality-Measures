This project aims to design and train a lightweight MLP model to identify the most influential nodes in a network **without requiring a full, computationally expensive ranking**. 

This Learning to Rank approach is designed to be as efficient as GNN-based models by utilizing only a single node feature: Node Degree. 

The goal is to achieve an ideal balance between implementation simplicity and ranking accuracy, providing a fast alternative for isolating key influential nodes in large-scale graphs.

#### **1. Model Architecture**

The model consists of a Multi-Layer Perceptron (MLP). It utilizes a structure with:

* **Input Layer:** Receives a single structural feature. The normalized node degree.
* **Hidden Layers:** For the current stage of this project, the model utilizes a single fully connected layer with ReLU activation to capture non-linear relationships between local topology and global centrality.
* **Output Layer:** A simple binary classification head using a Softmax function that predicts the probability of a node belonging to the *top-k* set.

#### **2. Methodology**

* **Input:** Node Degree normalized by the maximum degree within the graph.
* **Ground Truth Calculation:** We compute exact betweenness centrality scores for all nodes in the training set to provide the target labels for our MLP model.
* **Output:** Binary classification (Top-k or not).
* **Loss Function:** Cross-Entropy Loss, optimized using the ADAM optimizer.
* **Evaluation Metrics:**
  * **Ranking Accuracy:** Spearman’s Rank Correlation and Kendall’s Tau.
  * **Top-K Precision:** Precision@K to measure the overlap between predicted and true influential nodes.
  * **Efficiency:** Inference time in milliseconds (ms) and speedup ratio compared to exact algorithms.



#### **3. Training Data**

The model is trained on a diverse suite of **synthetic graph datasets** generated via:

* **Erdős-Rényi (ER):** For random connectivity patterns.
* **Barabási-Albert (BA):** To simulate scale-free properties and preferential attachment.
* **Newman-Watts-Strogatz (NWS):** To capture small-world characteristics.
* **Static Scale-Free (SSF):** For power-law degree distributions.

#### **4. Implementation Details**

* **Language:** Julia 1.11.7
* **Libraries:** All used libraries and their specific versions are recorded in the *Project.toml* and *Manifest.toml* files located in the project root.
* **Experiment Tracking:** All experiments, hyperparameter tuning, and training curves are tracked in real-time via Weights & Biases (WandB).
* **Current Results:** Current plots can be found here: https://api.wandb.ai/links/salmadoubali-student/pknx63xe

#### **5. Result Analysis**

The performance of this lightweight MLP approach will be benchmarked against GNN-based architectures, such as:

* **DRBC:** Directed Random Betweenness Centrality
* **ABCDE:** Approximating BC ranking with progressive-DropEdge.

The analysis focuses on whether a simple MLP, using only one local feature, can generalize across different graph scales and types with significantly lower computational overhead than GNN-based methods.
