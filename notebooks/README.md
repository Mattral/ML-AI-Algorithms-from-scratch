# scratchkit — Interactive Notebooks

Five detailed, end-to-end Jupyter notebooks covering every algorithm in
[`scratchkit`](https://pypi.org/project/scratchkit/) (`import mlscratch`).

Each notebook:
- Runs top-to-bottom without modification on **Google Colab** (free GPU optional, not required)
- Installs the package with a single `!pip install scratchkit` cell
- Uses real, publicly available datasets (Breast Cancer, Wine, Digits, Diabetes…)
- Explains the algorithm with the math, then shows the **running output** in the same cell
- Cross-checks results (accuracy, MSE, etc.) against quantitative references

---

## Run on Google Colab — click any badge

| Notebook | Algorithms covered | Open |
|---|---|---|
| `01_supervised_learning.ipynb` | Linear/Ridge/Lasso/ElasticNet Regression, Logistic Regression, KNN, Decision Trees, Random Forest, SVC (kernel SVM), Gradient Boosting, AdaBoost | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mattral/ML-AI-Algorithms-from-scratch/blob/main/notebooks/01_supervised_learning.ipynb) |
| `02_unsupervised_learning.ipynb` | KMeans, KMedoids, DBSCAN, Agglomerative Clustering, PCA, t-SNE, FastICA, Gaussian Mixture Model, Apriori | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mattral/ML-AI-Algorithms-from-scratch/blob/main/notebooks/02_unsupervised_learning.ipynb) |
| `03_bayesian_methods.ipynb` | Gaussian/Multinomial/Bernoulli Naive Bayes, Bayesian Linear Regression, Gaussian Process Regression, Hidden Markov Models, Kalman Filter, Bayesian Network, Bayesian Neural Network | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mattral/ML-AI-Algorithms-from-scratch/blob/main/notebooks/03_bayesian_methods.ipynb) |
| `04_reinforcement_learning.ipynb` | Q-Learning, Double Q-Learning, DQN, DDPG, TD3, PPO, SAC | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mattral/ML-AI-Algorithms-from-scratch/blob/main/notebooks/04_reinforcement_learning.ipynb) |
| `05_neural_networks.ipynb` | Perceptron, MLP, Autoencoders (vanilla/denoising/variational), RNN, LSTM, CNN, Transformer, GAN, Hopfield Network, RBM, RBF Network | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mattral/ML-AI-Algorithms-from-scratch/blob/main/notebooks/05_neural_networks.ipynb) |

---

## Run locally

```bash
pip install scratchkit "jupyter>=1.0" matplotlib scikit-learn scipy
jupyter notebook notebooks/01_supervised_learning.ipynb
```

---

## What each notebook teaches

### 01 — Supervised Learning
Every mainstream supervised algorithm on two real datasets (Breast Cancer Wisconsin for
classification, Diabetes for regression). Shows the **bias-variance trade-off** for
decision trees, **feature importances** from a random forest, the **kernel trick** for
SVM on a non-linearly-separable moon-shaped dataset, and a final head-to-head comparison
table of all classifiers on the same split.

### 02 — Unsupervised Learning
Real Wine dataset for clustering (with the **elbow method** to choose *k*, and a
comparison of K-Means vs K-Medoids vs DBSCAN vs GMM). The classic **ICA cocktail-party
problem** (two mixed waveforms separated from two microphone recordings). Handwritten
digits visualised with **t-SNE**. Market-basket **association rules** from grocery
transactions.

### 03 — Bayesian Methods
Three Naive Bayes flavours on three data types (continuous features, word counts, binary
presence/absence). **Bayesian linear regression** with calibrated uncertainty — the
posterior predictive interval visibly widens outside the training range. **Gaussian
Process Regression** with three different kernels showing how prior assumptions about
smoothness affect the fit. The **dishonest casino HMM** (rolling a fair vs loaded die —
Viterbi decodes the hidden die from rolls alone). **Kalman filter** tracking a noisy
position signal. The classic **Sprinkler Bayesian Network** with the famous
"explaining away" inference query.

### 04 — Reinforcement Learning
Tabular Q-Learning on the 4×4 GridWorld (learns to navigate to the goal and avoid the
pit, 100/100 greedy successes after 2000 training episodes). DQN, DDPG, TD3, PPO, and
SAC on custom continuous-action and discrete-action environments, with **learning
curves** and a final comparison plot of all three continuous-action algorithms (DDPG
vs TD3 vs SAC).

### 05 — Neural Networks
The full progression from a single-layer **Perceptron** (1957) to a **Multi-Layer
Perceptron** with backprop to a **CNN** on images (Digits resized to 16×16) to a
**Transformer encoder** with self-attention. Plus generative models: a **GAN** learning
to match a 2-D Gaussian distribution, and a **VAE** learning a structured latent space
for the Wine dataset. Biological/classical models: **Hopfield Network** (8×8 visual
pattern recovery from 20%-corrupted cues), **RBM** (latent feature discovery), and an
**RBF Network** comparing under- vs correctly-fitted regression.
