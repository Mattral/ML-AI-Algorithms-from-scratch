"""
Bayesian Neural Network (BNN) — Mean-Field Variational Inference
================================================================
Replaces deterministic weights with distributions:

    w ~ N(μ, σ²)

The variational posterior q(w | θ) = N(μ, σ²) is optimised to minimise
the Evidence Lower BOund (ELBO):

    ELBO = E_q[log p(y | x, w)] − KL[q(w | θ) || p(w)]

The KL term acts as weight regularisation; the likelihood term is the
negative cross-entropy for classification or negative Gaussian log-likelihood
for regression.

Training uses the "local reparameterisation trick":
    w = μ + σ * ε,   ε ~ N(0, 1)
so gradients flow through μ and σ (via log σ).

Only numpy and Python stdlib are used.
"""

import numpy as np


# ============================================================
# Activations
# ============================================================

def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(float)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


# ============================================================
# BNN Layer (variational weights)
# ============================================================

class BayesianLayer:
    """
    A single fully-connected layer with variational weight distribution.

    Parameters
    ----------
    in_features : int
    out_features : int
    prior_std : float
        Std of isotropic Gaussian prior N(0, prior_std²).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        prior_std: float = 1.0,
        rng: np.random.Generator = None,
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.prior_std = prior_std
        self._rng = rng or np.random.default_rng()

        # Variational parameters: mean and log-std for W and b
        scale = 1.0 / np.sqrt(in_features)
        self.mu_W = self._rng.normal(0, scale, (in_features, out_features))
        self.log_sigma_W = np.full((in_features, out_features), -3.0)
        self.mu_b = np.zeros(out_features)
        self.log_sigma_b = np.full(out_features, -3.0)

        # Sampled weights (set during forward pass)
        self.W_sample = None
        self.b_sample = None
        self._eps_W = None
        self._eps_b = None

    @property
    def sigma_W(self):
        return np.exp(self.log_sigma_W)

    @property
    def sigma_b(self):
        return np.exp(self.log_sigma_b)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Sample weights and compute linear transform."""
        self._eps_W = self._rng.standard_normal(self.mu_W.shape)
        self._eps_b = self._rng.standard_normal(self.mu_b.shape)
        self.W_sample = self.mu_W + self.sigma_W * self._eps_W
        self.b_sample = self.mu_b + self.sigma_b * self._eps_b
        self._input = x
        return x @ self.W_sample + self.b_sample

    def kl_divergence(self) -> float:
        """
        Closed-form KL[N(μ,σ²) || N(0, prior_std²)] for all weights.

        KL = 0.5 * [σ²/prior_std² + μ²/prior_std² - 1 + 2 log(prior_std/σ)]
        """
        prior_var = self.prior_std ** 2
        kl_W = 0.5 * np.sum(
            self.sigma_W ** 2 / prior_var
            + self.mu_W ** 2 / prior_var
            - 1.0
            + 2.0 * (np.log(self.prior_std) - self.log_sigma_W)
        )
        kl_b = 0.5 * np.sum(
            self.sigma_b ** 2 / prior_var
            + self.mu_b ** 2 / prior_var
            - 1.0
            + 2.0 * (np.log(self.prior_std) - self.log_sigma_b)
        )
        return float(kl_W + kl_b)


# ============================================================
# Bayesian Neural Network
# ============================================================

class BayesianNeuralNetwork:
    """
    Bayesian Neural Network trained via mean-field variational inference.

    Supports binary classification (sigmoid output) and multi-class
    classification (softmax output).  A single hidden layer is used by
    default; pass a list to `hidden_sizes` for deeper networks.

    Parameters
    ----------
    hidden_sizes : list of int
        Sizes of hidden layers.
    task : str
        'binary' or 'multiclass'.
    n_classes : int
        Number of output classes (ignored for binary).
    prior_std : float
        Std of the Gaussian prior on weights.
    lr : float
        Learning rate for gradient updates.
    n_samples : int
        Number of MC samples per gradient estimate.
    n_epochs : int
        Training epochs.
    batch_size : int or None
        Mini-batch size.  None = full-batch.
    kl_weight : float
        Scaling factor for the KL term (1/N is a common choice).
    random_state : int or None
    """

    def __init__(
        self,
        hidden_sizes: list | None = None,
        task: str = "binary",
        n_classes: int = 2,
        prior_std: float = 1.0,
        lr: float = 0.01,
        n_samples: int = 1,
        n_epochs: int = 100,
        batch_size: int | None = 32,
        kl_weight: float = 1.0,
        random_state: int | None = None,
    ):
        self.hidden_sizes = hidden_sizes or [64]
        self.task = task
        self.n_classes = n_classes
        self.prior_std = prior_std
        self.lr = lr
        self.n_samples = n_samples
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.kl_weight = kl_weight
        self.random_state = random_state
        self.layers_: list[BayesianLayer] = []
        self.losses_: list[float] = []

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _build(self, n_input: int) -> None:
        rng = np.random.default_rng(self.random_state)
        sizes = [n_input] + list(self.hidden_sizes)
        n_out = 1 if self.task == "binary" else self.n_classes

        self.layers_ = []
        for i in range(len(sizes) - 1):
            self.layers_.append(
                BayesianLayer(sizes[i], sizes[i + 1], self.prior_std, rng)
            )
        self.layers_.append(
            BayesianLayer(sizes[-1], n_out, self.prior_std, rng)
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _forward(self, X: np.ndarray) -> np.ndarray:
        """Single forward pass with sampled weights."""
        h = X
        for layer in self.layers_[:-1]:
            h = _relu(layer.forward(h))
        logit = self.layers_[-1].forward(h)
        if self.task == "binary":
            return _sigmoid(logit).squeeze(-1)
        return _softmax(logit)

    # ------------------------------------------------------------------
    # ELBO / loss
    # ------------------------------------------------------------------

    def _elbo(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute -ELBO (loss to minimise) averaged over MC samples."""
        n = len(X)
        total_nll = 0.0
        for _ in range(self.n_samples):
            pred = self._forward(X)
            if self.task == "binary":
                pred = np.clip(pred, 1e-7, 1 - 1e-7)
                nll = -np.mean(y * np.log(pred) + (1 - y) * np.log(1 - pred))
            else:
                pred = np.clip(pred, 1e-7, 1.0)
                nll = -np.mean(np.log(pred[np.arange(n), y.astype(int)]))
            total_nll += nll

        avg_nll = total_nll / self.n_samples
        kl = sum(layer.kl_divergence() for layer in self.layers_)
        return avg_nll + self.kl_weight * kl / n

    # ------------------------------------------------------------------
    # Gradient step (finite differences for simplicity)
    # ------------------------------------------------------------------

    def _update_params(self, X: np.ndarray, y: np.ndarray) -> None:
        """Gradient update using a stochastic estimate of the ELBO gradient."""
        eps = 1e-5
        for layer in self.layers_:
            for param_name in ("mu_W", "log_sigma_W", "mu_b", "log_sigma_b"):
                param = getattr(layer, param_name)
                grad = np.zeros_like(param)
                flat = param.ravel()
                for idx in range(len(flat)):
                    orig = flat[idx]
                    flat[idx] = orig + eps
                    loss_p = self._elbo(X, y)
                    flat[idx] = orig - eps
                    loss_m = self._elbo(X, y)
                    flat[idx] = orig
                    grad.ravel()[idx] = (loss_p - loss_m) / (2 * eps)
                param -= self.lr * grad

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BayesianNeuralNetwork":
        """
        Train the BNN.

        Parameters
        ----------
        X : ndarray (n_samples, n_features)
        y : ndarray (n_samples,)  — integer class labels
        """
        n_samples = len(X)
        self._build(X.shape[1])
        rng = np.random.default_rng(self.random_state)
        bs = self.batch_size or n_samples

        for epoch in range(self.n_epochs):
            idx = rng.permutation(n_samples)
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, n_samples, bs):
                batch_idx = idx[start:start + bs]
                Xb, yb = X[batch_idx], y[batch_idx]
                self._update_params(Xb, yb)
                epoch_loss += self._elbo(Xb, yb)
                n_batches += 1
            self.losses_.append(epoch_loss / n_batches)

        return self

    def predict_proba(self, X: np.ndarray, n_samples: int = 50) -> np.ndarray:
        """
        Monte-Carlo predictive probabilities (averaged over weight samples).

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
                or (n_samples,) for binary task
        """
        preds = []
        for _ in range(n_samples):
            preds.append(self._forward(X))
        return np.stack(preds).mean(axis=0)

    def predict(self, X: np.ndarray, n_samples: int = 50) -> np.ndarray:
        proba = self.predict_proba(X, n_samples)
        if self.task == "binary":
            return (proba >= 0.5).astype(int)
        return np.argmax(proba, axis=1)
