"""
Deep learning models for the Lifecycle Intelligence System: MLP, LSTM and
Transformer, implemented in PyTorch with explicit epoch-wise training.

Why these architectures, on tabular data
----------------------------------------
LSTM and Transformer are sequence models, and a material record is not
naturally a sequence. Applying them here requires an explicit framing, and
stating it honestly matters more than hiding it:

    Each of the F input features is treated as one TOKEN in a sequence of
    length F. A per-feature linear projection maps each scalar feature value
    to a d_model-dimensional embedding (the numerical-embedding scheme used by
    FT-Transformer, Gorishniy et al., 2021). The sequence of F feature-tokens
    is then processed by the recurrent or attention stack.

Under this framing:
  * the LSTM reads the feature-tokens in order and carries state across them,
    so it models feature interactions through its gating;
  * the Transformer applies self-attention across feature-tokens, so every
    feature can attend to every other directly - which is the mechanism that
    makes attention attractive for tabular interactions.

This is a legitimate, published framing, but it carries a real caveat worth
stating in the report: feature ORDER is arbitrary, so the LSTM's sequential
inductive bias has no physical meaning here, whereas the Transformer's
attention is permutation-equivariant and therefore a better match. The results
bear this out, and that contrast is itself a finding worth presenting.

Activation functions (evaluator requirement)
--------------------------------------------
Every activation is chosen explicitly rather than left at a library default:
  * MLP hidden layers   - ReLU by default, selectable among ReLU / GELU /
                          Tanh / LeakyReLU via the `activation` parameter
  * LSTM gates          - Sigmoid on the input/forget/output gates and Tanh on
                          the cell state. These are intrinsic to the LSTM cell
                          definition and are not free parameters.
  * LSTM head           - the selected `activation`
  * Transformer FFN     - GELU, the standard choice in modern encoder blocks
  * Output layer        - Softmax, applied inside CrossEntropyLoss for
                          numerical stability (log-sum-exp), and explicitly
                          when probabilities are needed at predict time

Training
--------
Explicit epoch loop. After every epoch the model is evaluated on the validation
split, recording validation loss and validation accuracy. Early stopping
monitors VALIDATION LOSS and restores the weights from the best epoch, so the
reported model is the one that minimised validation loss rather than the one
left at the final epoch.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "leaky_relu": nn.LeakyReLU,
}


def _set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


class FeatureTokenizer(nn.Module):
    """
    Map a tabular row of F scalars to a sequence of F embeddings.

    Each feature gets its own learned weight vector and bias, so feature j is
    embedded as  x_j * w_j + b_j  in R^d. This is the numerical-embedding step
    from FT-Transformer; without it, a sequence model would see a sequence of
    bare scalars with no way to tell which feature it was looking at.
    """

    def __init__(self, n_features, d_model):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, d_model))
        self.bias = nn.Parameter(torch.empty(n_features, d_model))
        nn.init.normal_(self.weight, std=0.02)
        nn.init.zeros_(self.bias)

    def forward(self, x):                      # x: (B, F)
        return x.unsqueeze(-1) * self.weight + self.bias   # (B, F, d_model)


class MLPNet(nn.Module):
    """Feed-forward baseline. Activation is explicit and configurable."""

    def __init__(self, n_features, n_classes, hidden=(128, 64),
                 activation="relu", dropout=0.2):
        super().__init__()
        act = ACTIVATIONS[activation]
        layers, prev = [], n_features
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), act(),
                       nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class LSTMNet(nn.Module):
    """
    Bidirectional LSTM over the feature-token sequence.

    The LSTM cell's own Sigmoid gates and Tanh cell activation are intrinsic to
    its definition; the configurable `activation` applies to the classification
    head that sits on top of the pooled hidden state.
    """

    def __init__(self, n_features, n_classes, d_model=64, hidden=64,
                 num_layers=2, activation="tanh", dropout=0.2):
        super().__init__()
        act = ACTIVATIONS[activation]
        self.tokenizer = FeatureTokenizer(n_features, d_model)
        self.lstm = nn.LSTM(
            input_size=d_model, hidden_size=hidden, num_layers=num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden * 2), nn.Linear(hidden * 2, hidden), act(),
            nn.Dropout(dropout), nn.Linear(hidden, n_classes))

    def forward(self, x):
        tokens = self.tokenizer(x)             # (B, F, d_model)
        out, _ = self.lstm(tokens)             # (B, F, 2*hidden)
        # Mean-pool across feature-tokens rather than taking the last step:
        # feature order is arbitrary, so no single position is privileged.
        return self.head(out.mean(dim=1))


class TransformerNet(nn.Module):
    """
    Transformer encoder over feature-tokens, with a prepended [CLS] token.

    Self-attention lets every feature attend to every other directly, which is
    the natural way to model the feature interactions this problem contains -
    for example hazard level interacting with contamination and reprocessability.
    No positional encoding is added: feature order is arbitrary, so the
    permutation-equivariance of plain attention is exactly what is wanted here.
    """

    def __init__(self, n_features, n_classes, d_model=64, nhead=8,
                 num_layers=3, dim_feedforward=128, activation="gelu",
                 dropout=0.1):
        super().__init__()
        self.tokenizer = FeatureTokenizer(n_features, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation=activation, batch_first=True,
            norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, n_classes))

    def forward(self, x):
        tokens = self.tokenizer(x)                             # (B, F, d)
        cls = self.cls_token.expand(tokens.size(0), -1, -1)    # (B, 1, d)
        seq = torch.cat([cls, tokens], dim=1)                  # (B, F+1, d)
        encoded = self.encoder(seq)
        return self.head(encoded[:, 0])        # read the [CLS] position


class TorchTabularClassifier(ClassifierMixin, BaseEstimator):
    """
    scikit-learn compatible wrapper around the three architectures above.

    Exposes `history_`, a per-epoch record of training loss, validation loss
    and validation accuracy, which is what the evaluator's requirement to
    "focus more on validation loss and validation accuracy" needs in order to
    be plotted.
    """

    _estimator_type = "classifier"

    def __init__(self, arch="mlp", d_model=64, hidden=64, num_layers=2,
                 activation="relu", dropout=0.2, lr=1e-3, weight_decay=1e-4,
                 epochs=60, batch_size=128, patience=12, balanced=True,
                 random_state=42, verbose=False):
        self.arch = arch
        self.d_model = d_model
        self.hidden = hidden
        self.num_layers = num_layers
        self.activation = activation
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.balanced = balanced
        self.random_state = random_state
        self.verbose = verbose

    # -- construction -----------------------------------------------------
    def _build(self, n_features, n_classes):
        if self.arch == "mlp":
            hidden = (self.hidden * 2, self.hidden)
            return MLPNet(n_features, n_classes, hidden=hidden,
                          activation=self.activation, dropout=self.dropout)
        if self.arch == "lstm":
            return LSTMNet(n_features, n_classes, d_model=self.d_model,
                           hidden=self.hidden, num_layers=self.num_layers,
                           activation=self.activation, dropout=self.dropout)
        if self.arch == "transformer":
            # nhead must divide d_model exactly.
            nhead = 8 if self.d_model % 8 == 0 else 4
            return TransformerNet(
                n_features, n_classes, d_model=self.d_model, nhead=nhead,
                num_layers=self.num_layers, dim_feedforward=self.d_model * 2,
                activation="gelu", dropout=self.dropout)
        raise ValueError(f"unknown arch: {self.arch}")

    # -- training ---------------------------------------------------------
    def fit(self, X, y, eval_set=None):
        """
        Train with an explicit epoch loop.

        `eval_set` is a (X_val, y_val) tuple of ALREADY-PREPROCESSED arrays.
        When supplied, that split drives both the per-epoch curves and early
        stopping. When it is absent - as during a GridSearchCV inner fit - a
        10% slice of the training data stands in, so the model can still early
        stop without ever touching the real validation split.
        """
        _set_seed(self.random_state)
        X = np.asarray(X, dtype=np.float32)
        self._le = LabelEncoder().fit(y)
        self.classes_ = self._le.classes_
        y_enc = self._le.transform(y)
        n_classes = len(self.classes_)

        if eval_set is not None:
            X_val = np.asarray(eval_set[0], dtype=np.float32)
            y_val = self._le.transform(eval_set[1])
        else:
            rng = np.random.default_rng(self.random_state)
            idx = rng.permutation(len(X))
            cut = max(int(0.10 * len(X)), 1)
            val_idx, tr_idx = idx[:cut], idx[cut:]
            X_val, y_val = X[val_idx], y_enc[val_idx]
            X, y_enc = X[tr_idx], y_enc[tr_idx]

        self.model_ = self._build(X.shape[1], n_classes).to(DEVICE)

        # Class weights counter the Dispose/Reuse minority imbalance, the same
        # correction class_weight='balanced' applies to the sklearn models.
        if self.balanced:
            cw = compute_class_weight("balanced", classes=np.arange(n_classes),
                                      y=y_enc)
            weight = torch.tensor(cw, dtype=torch.float32, device=DEVICE)
        else:
            weight = None
        criterion = nn.CrossEntropyLoss(weight=weight)   # applies log-softmax

        optimizer = torch.optim.AdamW(self.model_.parameters(), lr=self.lr,
                                      weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=max(self.patience // 3, 2))

        Xt = torch.tensor(X, device=DEVICE)
        yt = torch.tensor(y_enc, dtype=torch.long, device=DEVICE)
        Xv = torch.tensor(X_val, device=DEVICE)
        yv = torch.tensor(y_val, dtype=torch.long, device=DEVICE)

        n = len(Xt)
        best_val_loss, best_state, bad_epochs = float("inf"), None, 0
        history = {"epoch": [], "train_loss": [], "val_loss": [],
                   "train_acc": [], "val_acc": [], "lr": []}

        for epoch in range(1, self.epochs + 1):
            self.model_.train()
            perm = torch.randperm(n, device=DEVICE)
            total_loss, total_correct = 0.0, 0

            for start in range(0, n, self.batch_size):
                batch = perm[start:start + self.batch_size]
                if len(batch) < 2:        # BatchNorm needs at least 2 samples
                    continue
                xb, yb = Xt[batch], yt[batch]
                optimizer.zero_grad()
                logits = self.model_(xb)
                loss = criterion(logits, yb)
                loss.backward()
                # Gradient clipping: the LSTM in particular is prone to
                # exploding gradients over a 36-step feature sequence.
                nn.utils.clip_grad_norm_(self.model_.parameters(), 5.0)
                optimizer.step()
                total_loss += loss.item() * len(batch)
                total_correct += (logits.argmax(1) == yb).sum().item()

            train_loss = total_loss / n
            train_acc = total_correct / n

            self.model_.eval()
            with torch.no_grad():
                val_logits = self.model_(Xv)
                val_loss = criterion(val_logits, yv).item()
                val_acc = (val_logits.argmax(1) == yv).float().mean().item()

            scheduler.step(val_loss)
            history["epoch"].append(epoch)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)
            history["lr"].append(optimizer.param_groups[0]["lr"])

            if self.verbose and (epoch % 10 == 0 or epoch == 1):
                print(f"    epoch {epoch:3d}  train_loss={train_loss:.4f}  "
                      f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

            # Early stopping on VALIDATION LOSS, keeping the best weights.
            if val_loss < best_val_loss - 1e-5:
                best_val_loss = val_loss
                best_state = {k: v.detach().clone()
                              for k, v in self.model_.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    if self.verbose:
                        print(f"    early stop at epoch {epoch} "
                              f"(best val_loss={best_val_loss:.4f})")
                    break

        if best_state is not None:
            self.model_.load_state_dict(best_state)

        self.history_ = history
        self.best_epoch_ = int(np.argmin(history["val_loss"]) + 1)
        self.best_val_loss_ = float(np.min(history["val_loss"]))
        self.best_val_acc_ = float(history["val_acc"][self.best_epoch_ - 1])
        return self

    # -- inference --------------------------------------------------------
    def _logits(self, X):
        self.model_.eval()
        X = torch.tensor(np.asarray(X, dtype=np.float32), device=DEVICE)
        outs = []
        with torch.no_grad():
            for start in range(0, len(X), 1024):
                outs.append(self.model_(X[start:start + 1024]).cpu())
        return torch.cat(outs)

    def predict_proba(self, X):
        # Explicit Softmax at inference; during training it is folded into
        # CrossEntropyLoss for numerical stability.
        return torch.softmax(self._logits(X), dim=1).numpy()

    def predict(self, X):
        return self._le.inverse_transform(self._logits(X).argmax(1).numpy())
