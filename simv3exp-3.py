"""SouDevail. Experiment 3. Deep refactor with Hybrid Adam Self-Healing."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import copy
import math

torch.manual_seed(0)

target_loss = 0.45
max_rollbacks = 3
stabilization_window = 3
lr_dampening = 0.5
beta1_damping = 0.85
exp_avg_decay = 0.3
epochs = 50
batch_size = 64
d_in, d_compact, d_common, d_out = 64, 6, 8, 10
n_macro, k_macro, n_nano, k_nano, tau = 1000, 50, 5, 2, 1.0


def generate_synthetic_data(n_samples, noise_level=0.1):
    X = torch.randn(n_samples, d_in)
    W_true = torch.randn(d_in, d_out) * 0.5
    noise = torch.randn(n_samples, d_out) * noise_level
    y = X @ W_true + (X**2).sum(dim=-1, keepdim=True) * 0.05 + noise
    y = (y - y.mean(0)) / (y.std(0) + 1e-8)
    return X, y


def gumbel_topk(logits, k, training=True):
    if training:
        g = -torch.log(-torch.log(torch.rand_like(logits) + 1e-9) + 1e-9)
        scores = (logits + g) / tau
    else:
        scores = logits
    topk = scores.topk(k, dim=-1)
    mask = torch.zeros_like(logits).scatter_(-1, topk.indices, 1.0)
    w = F.softmax(topk.values, dim=-1)
    w_full = torch.zeros_like(logits).scatter_(-1, topk.indices, w)
    return mask, w_full, topk.indices


class DFSHE(nn.Module):
    def __init__(self):
        super().__init__()
        self.compress = nn.Linear(d_in, d_compact)
        self.macro_router = nn.Linear(d_compact, n_macro)
        self.micro_w = nn.Parameter(torch.randn(n_macro, n_nano, d_compact) * 0.1)
        self.micro_b = nn.Parameter(torch.zeros(n_macro, n_nano))
        self.mlp0_w = nn.Parameter(torch.randn(n_macro, d_compact, d_compact) * 0.1)
        self.mlp0_b = nn.Parameter(torch.zeros(n_macro, d_compact))
        self.cnn_w = nn.Parameter(torch.randn(n_macro, 2, 3) * 0.1)
        self.cnn_b = nn.Parameter(torch.zeros(n_macro, 2))
        self.cnn_proj_w = nn.Parameter(torch.randn(n_macro, d_compact, 2 * d_compact) * 0.1)
        self.cnn_proj_b = nn.Parameter(torch.zeros(n_macro, d_compact))
        self.attn_q = nn.Parameter(torch.randn(n_macro, d_compact, d_compact) * 0.1)
        self.attn_k = nn.Parameter(torch.randn(n_macro, d_compact, d_compact) * 0.1)
        self.attn_v = nn.Parameter(torch.randn(n_macro, d_compact, d_compact) * 0.1)
        self.tree_gate_w = nn.Parameter(torch.randn(n_macro, 2, d_compact) * 0.1)
        self.tree_gate_b = nn.Parameter(torch.zeros(n_macro, 2))
        self.tree_leaf_w = nn.Parameter(torch.randn(n_macro, 2, d_compact, d_compact) * 0.1)
        self.rnn_ih = nn.Parameter(torch.randn(n_macro, d_compact, d_compact) * 0.1)
        self.rnn_hh = nn.Parameter(torch.randn(n_macro, d_compact, d_compact) * 0.1)
        self.rnn_b_ih = nn.Parameter(torch.zeros(n_macro, d_compact))
        self.rnn_b_hh = nn.Parameter(torch.zeros(n_macro, d_compact))
        self.bridge_w = nn.Parameter(torch.randn(n_macro, d_common, d_compact) * 0.1)
        self.bridge_b = nn.Parameter(torch.zeros(n_macro, d_common))
        self.head = nn.Linear(d_common, d_out)

    def forward(self, x):
        B = x.size(0)
        h = self.compress(x)
        logits_m = self.macro_router(h)
        mask_m, w_m, idx_m = gumbel_topk(logits_m, k_macro, self.training)
        w_m_sel = torch.gather(w_m, 1, idx_m)
        idx = idx_m.reshape(-1)
        h_b = h.unsqueeze(1).expand(B, k_macro, d_compact).reshape(B * k_macro, d_compact)
        #microteams optimisation
        logits_micro = torch.einsum('bd,bnd->bn', h_b, self.micro_w[idx]) + self.micro_b[idx]
        logits_micro = logits_micro.view(B, k_macro, n_nano)
        _, w_micro, _ = gumbel_topk(logits_micro, k_nano, self.training)
        #nanoexperts: mlp / conv / lin-attn / soft-tree / elman-rnn
        mlp0 = F.relu(torch.einsum('bd,bcd->bc', h_b, self.mlp0_w[idx]) + self.mlp0_b[idx])
        x_pad = F.pad(h_b.unsqueeze(1), (1, 1)).unfold(-1, 3, 1).squeeze(1)
        cnn_c = torch.einsum('bij,bkj->bik', self.cnn_w[idx], x_pad) + self.cnn_b[idx].unsqueeze(-1)
        cnn = torch.einsum('bd,bcd->bc', cnn_c.flatten(1), self.cnn_proj_w[idx]) + self.cnn_proj_b[idx]
        q = torch.einsum('bd,bcd->bc', h_b, self.attn_q[idx])
        kk = torch.einsum('bd,bcd->bc', h_b, self.attn_k[idx])
        v = torch.einsum('bd,bcd->bc', h_b, self.attn_v[idx])
        attn = q * ((kk * v).sum(-1, keepdim=True)) / d_compact
        gate = F.softmax(torch.einsum('bd,bnd->bn', h_b, self.tree_gate_w[idx]) + self.tree_gate_b[idx], dim=-1)
        leaf = torch.einsum('bijd,bd->bij', self.tree_leaf_w[idx], h_b)
        tree = (gate.unsqueeze(-1) * leaf).sum(1)
        h1 = torch.tanh(torch.einsum('bd,bcd->bc', h_b, self.rnn_ih[idx]) + self.rnn_b_ih[idx])
        rnn = torch.tanh(torch.einsum('bd,bcd->bc', h1, self.rnn_hh[idx]) + self.rnn_b_hh[idx])
        nano = torch.stack([mlp0, cnn, attn, tree, rnn], dim=1).view(B, k_macro, n_nano, d_compact)
        routed = (nano * w_micro.unsqueeze(-1)).sum(2).view(B * k_macro, d_compact)
        bridge = torch.einsum('bd,bcd->bc', routed, self.bridge_w[idx]) + self.bridge_b[idx]
        bridge = bridge.view(B, k_macro, d_common)
        out = (bridge * w_m_sel.unsqueeze(-1)).sum(1)
        return self.head(out), mask_m


class SelfHealing:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.best_state = None
        self.best_loss = float('inf')
        self.loss_history = []
        self.consecutive_rises = 0
        self.rollback_count = 0

    def _adam_surgery(self):
        # hybrid: damp betas[0] + partial exp_avg reset on routing params
        for group in self.optimizer.param_groups:
            b1, b2 = group['betas']
            group['betas'] = (max(b1 * beta1_damping, 0.5), b2)
        routing_params = [
            self.model.compress.weight, self.model.compress.bias,
            self.model.macro_router.weight, self.model.macro_router.bias,
        ]
        for p in routing_params:
            if p in self.optimizer.state and 'exp_avg' in self.optimizer.state[p]:
                self.optimizer.state[p]['exp_avg'].mul_(exp_avg_decay)

    def step(self, loss_val):
        if not math.isfinite(loss_val):
            if self.best_state:
                self.model.load_state_dict(self.best_state)
                self.optimizer.zero_grad()
                self.rollback_count += 1
                return self.rollback_count <= max_rollbacks
            return False

        self.loss_history.append(loss_val)
        if len(self.loss_history) > stabilization_window:
            self.loss_history.pop(0)

        if len(self.loss_history) >= 2:
            if self.loss_history[-1] > self.loss_history[-2]:
                self.consecutive_rises += 1
            else:
                self.consecutive_rises = 0
            if self.consecutive_rises >= 3:
                if self.best_state:
                    self.model.load_state_dict(self.best_state)
                    self.optimizer.zero_grad()
                    self.rollback_count += 1
                    self._adam_surgery()
                    self.consecutive_rises = 0
                    return self.rollback_count <= max_rollbacks
                return False

        if loss_val < self.best_loss:
            self.best_loss = loss_val
            self.best_state = copy.deepcopy(self.model.state_dict())
        return True


def run_training():
    start_time = time.time()
    X, y = generate_synthetic_data(2000)
    X, y = X.float(), y.float()

    n_train = 1600
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:], y[n_train:]

    model = DFSHE()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    criterion = nn.MSELoss()
    controller = SelfHealing(model, optimizer)

    print("epoch,train_loss,val_loss,rollback_count,best_loss,beta1")

    val_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train)
        epoch_loss, n_batches = 0.0, 0
        for i in range(0, n_train, batch_size):
            xb, yb = X_train[perm[i:i + batch_size]], y_train[perm[i:i + batch_size]]
            optimizer.zero_grad()
            out, _ = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        avg_train = epoch_loss / n_batches

        model.eval()
        with torch.no_grad():
            val_out, _ = model(X_val)
            val_loss = criterion(val_out, y_val).item()
        model.train()

        b1 = optimizer.param_groups[0]['betas'][0]

        if not controller.step(val_loss):
            print(f"{epoch+1},{avg_train:.6f},{val_loss:.6f},{controller.rollback_count},{controller.best_loss:.6f},{b1:.4f}")
            print("TRAINING_STOPPED")
            break
        elif val_loss < target_loss:
            print(f"{epoch+1},{avg_train:.6f},{val_loss:.6f},{controller.rollback_count},{controller.best_loss:.6f},{b1:.4f}")
            print("TARGET_REACHED")
            break
        elif (epoch + 1) % 10 == 0:
            print(f"{epoch+1},{avg_train:.6f},{val_loss:.6f},{controller.rollback_count},{controller.best_loss:.6f},{b1:.4f}")

    total_time = time.time() - start_time
    print(f"TOTAL_TIME,{total_time:.3f}")
    print(f"FINAL_STATUS,{val_loss < target_loss}")


if __name__ == "__main__":
    run_training()
