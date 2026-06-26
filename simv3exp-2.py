"""SouDevail. Experiment 2. Normalized mini-batch training loop."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import copy
import math

torch.manual_seed(0)

class Config:
    TARGET_LOSS = 0.45
    MAX_ROLLBACKS = 3
    STABILIZATION_WINDOW = 3
    LR_DAMPENING = 0.5
    MOMENTUM_BOOST = 1.3
    EPOCHS = 50
    BATCH_SIZE = 64
    D_IN, D_COMPACT, D_COMMON, D_OUT = 64, 6, 8, 10
    N_MACRO, K_MACRO, N_NANO, K_NANO, TAU = 1000, 50, 5, 2, 1.0

cfg = Config()


def generate_synthetic_data(n_samples, d_in, d_out, noise_level=0.1):
    X = torch.randn(n_samples, d_in)
    W_true = torch.randn(d_in, d_out) * 0.5
    noise = torch.randn(n_samples, d_out) * noise_level
    y = X @ W_true + (X**2).sum(dim=-1, keepdim=True) * 0.05 + noise
    # fix-1: normalise targets per-dim; MSE baseline now ~1.0 not ~23
    y = (y - y.mean(0)) / (y.std(0) + 1e-8)
    return X, y


def gumbel_topk(logits, k, tau=1.0, training=True):
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
        self.compress = nn.Linear(cfg.D_IN, cfg.D_COMPACT)
        self.macro_router = nn.Linear(cfg.D_COMPACT, cfg.N_MACRO)
        self.micro_w = nn.Parameter(torch.randn(cfg.N_MACRO, cfg.N_NANO, cfg.D_COMPACT) * 0.1)
        self.micro_b = nn.Parameter(torch.zeros(cfg.N_MACRO, cfg.N_NANO))
        self.mlp0_w = nn.Parameter(torch.randn(cfg.N_MACRO, cfg.D_COMPACT, cfg.D_COMPACT) * 0.1)
        self.mlp0_b = nn.Parameter(torch.zeros(cfg.N_MACRO, cfg.D_COMPACT))
        self.cnn_w = nn.Parameter(torch.randn(cfg.N_MACRO, 2, 3) * 0.1)
        self.cnn_b = nn.Parameter(torch.zeros(cfg.N_MACRO, 2))
        self.cnn_proj_w = nn.Parameter(torch.randn(cfg.N_MACRO, cfg.D_COMPACT, 2 * cfg.D_COMPACT) * 0.1)
        self.cnn_proj_b = nn.Parameter(torch.zeros(cfg.N_MACRO, cfg.D_COMPACT))
        self.attn_q = nn.Parameter(torch.randn(cfg.N_MACRO, cfg.D_COMPACT, cfg.D_COMPACT) * 0.1)
        self.attn_k = nn.Parameter(torch.randn(cfg.N_MACRO, cfg.D_COMPACT, cfg.D_COMPACT) * 0.1)
        self.attn_v = nn.Parameter(torch.randn(cfg.N_MACRO, cfg.D_COMPACT, cfg.D_COMPACT) * 0.1)
        self.tree_gate_w = nn.Parameter(torch.randn(cfg.N_MACRO, 2, cfg.D_COMPACT) * 0.1)
        self.tree_gate_b = nn.Parameter(torch.zeros(cfg.N_MACRO, 2))
        self.tree_leaf_w = nn.Parameter(torch.randn(cfg.N_MACRO, 2, cfg.D_COMPACT, cfg.D_COMPACT) * 0.1)
        self.rnn_ih = nn.Parameter(torch.randn(cfg.N_MACRO, cfg.D_COMPACT, cfg.D_COMPACT) * 0.1)
        self.rnn_hh = nn.Parameter(torch.randn(cfg.N_MACRO, cfg.D_COMPACT, cfg.D_COMPACT) * 0.1)
        self.rnn_b = nn.Parameter(torch.zeros(cfg.N_MACRO, cfg.D_COMPACT))
        self.bridge_w = nn.Parameter(torch.randn(cfg.N_MACRO, cfg.D_COMMON, cfg.D_COMPACT) * 0.1)
        self.bridge_b = nn.Parameter(torch.zeros(cfg.N_MACRO, cfg.D_COMMON))
        self.head = nn.Linear(cfg.D_COMMON, cfg.D_OUT)

    def forward(self, x):
        B = x.size(0)
        h = self.compress(x)
        logits_m = self.macro_router(h)
        mask_m, w_m, idx_m = gumbel_topk(logits_m, cfg.K_MACRO, cfg.TAU, self.training)
        w_m_sel = torch.gather(w_m, 1, idx_m)
        idx = idx_m.reshape(-1)
        h_b = h.unsqueeze(1).expand(B, cfg.K_MACRO, cfg.D_COMPACT).reshape(B * cfg.K_MACRO, cfg.D_COMPACT)
        #microteams optimisation
        logits_micro = torch.einsum('bd,bnd->bn', h_b, self.micro_w[idx]) + self.micro_b[idx]
        logits_micro = logits_micro.view(B, cfg.K_MACRO, cfg.N_NANO)
        _, w_micro, _ = gumbel_topk(logits_micro, cfg.K_NANO, cfg.TAU, self.training)  # fix-2: K_NANO not N_NANO
        #nanoexperts: mlp / conv / lin-attn / soft-tree / elman-rnn
        mlp0 = F.relu(torch.einsum('bd,bcd->bc', h_b, self.mlp0_w[idx]) + self.mlp0_b[idx])
        x_pad = F.pad(h_b.unsqueeze(1), (1, 1)).unfold(-1, 3, 1).squeeze(1)
        cnn_c = torch.einsum('bij,bkj->bik', self.cnn_w[idx], x_pad) + self.cnn_b[idx].unsqueeze(-1)
        cnn = torch.einsum('bd,bcd->bc', cnn_c.flatten(1), self.cnn_proj_w[idx]) + self.cnn_proj_b[idx]
        q = torch.einsum('bd,bcd->bc', h_b, self.attn_q[idx])
        kk = torch.einsum('bd,bcd->bc', h_b, self.attn_k[idx])
        v = torch.einsum('bd,bcd->bc', h_b, self.attn_v[idx])
        attn = q * ((kk * v).sum(-1, keepdim=True)) / cfg.D_COMPACT
        gate = F.softmax(torch.einsum('bd,bnd->bn', h_b, self.tree_gate_w[idx]) + self.tree_gate_b[idx], dim=-1)
        leaf = torch.einsum('bijd,bd->bij', self.tree_leaf_w[idx], h_b)
        tree = (gate.unsqueeze(-1) * leaf).sum(1)
        h1 = torch.tanh(torch.einsum('bd,bcd->bc', h_b, self.rnn_ih[idx]) + self.rnn_b[idx])
        rnn = torch.tanh(torch.einsum('bd,bcd->bc', h1, self.rnn_hh[idx]) + self.rnn_b[idx])
        nano = torch.stack([mlp0, cnn, attn, tree, rnn], dim=1).view(B, cfg.K_MACRO, cfg.N_NANO, cfg.D_COMPACT)
        routed = (nano * w_micro.unsqueeze(-1)).sum(2).view(B * cfg.K_MACRO, cfg.D_COMPACT)
        bridge = torch.einsum('bd,bcd->bc', routed, self.bridge_w[idx]) + self.bridge_b[idx]
        bridge = bridge.view(B, cfg.K_MACRO, cfg.D_COMMON)
        out = (bridge * w_m_sel.unsqueeze(-1)).sum(1)
        return self.head(out), mask_m


class SimulacrumV3:
    def __init__(self, model, target_loss=0.45, max_rollbacks=3):
        self.model = model
        self.target_loss = target_loss
        self.max_rollbacks = max_rollbacks
        self.best_state = None
        self.best_loss = float('inf')
        self.loss_history = []
        self.consecutive_rises = 0
        self.rollback_count = 0
        self.stabilization_window = cfg.STABILIZATION_WINDOW

    def init_optimizer(self, optimizer):
        self.optimizer = optimizer
        for group in optimizer.param_groups:
            self.initial_lr = group['lr']
            self.initial_momentum = group.get('momentum', 0)

    def step(self, loss_val):
        if not math.isfinite(loss_val):
            if self.best_state:
                self.model.load_state_dict(self.best_state)
                self.optimizer.zero_grad()
                self.rollback_count += 1
                return self.rollback_count <= self.max_rollbacks
            return False

        self.loss_history.append(loss_val)
        if len(self.loss_history) > self.stabilization_window:
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
                    for group in self.optimizer.param_groups:
                        group['lr'] = max(group['lr'] * cfg.LR_DAMPENING, 1e-5)
                        group['momentum'] = min(group.get('momentum', 0) + cfg.MOMENTUM_BOOST, 0.99)
                    self.consecutive_rises = 0
                    return self.rollback_count <= self.max_rollbacks
                return False

        if loss_val < self.best_loss:
            self.best_loss = loss_val
            self.best_state = copy.deepcopy(self.model.state_dict())
        return True


def run_training():
    start_time = time.time()
    X, y = generate_synthetic_data(2000, cfg.D_IN, cfg.D_OUT)
    X, y = X.float(), y.float()

    # fix-3: mini-batch + 80/20 train/val; controller tracks val_loss
    n_train = 1600
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:], y[n_train:]

    model = DFSHE()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    criterion = nn.MSELoss()
    controller = SimulacrumV3(model, target_loss=cfg.TARGET_LOSS, max_rollbacks=cfg.MAX_ROLLBACKS)
    controller.init_optimizer(optimizer)

    print("epoch,train_loss,val_loss,rollback_count,best_loss")

    val_loss = float('inf')
    for epoch in range(cfg.EPOCHS):
        model.train()
        perm = torch.randperm(n_train)
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, n_train, cfg.BATCH_SIZE):
            batch_idx = perm[i:i + cfg.BATCH_SIZE]
            xb, yb = X_train[batch_idx], y_train[batch_idx]
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

        if not controller.step(val_loss):
            print(f"{epoch+1},{avg_train:.6f},{val_loss:.6f},{controller.rollback_count},{controller.best_loss:.6f}")
            print("TRAINING_STOPPED")
            break

        if (epoch + 1) % 10 == 0:
            print(f"{epoch+1},{avg_train:.6f},{val_loss:.6f},{controller.rollback_count},{controller.best_loss:.6f}")

        if val_loss < cfg.TARGET_LOSS:
            print(f"{epoch+1},{avg_train:.6f},{val_loss:.6f},{controller.rollback_count},{controller.best_loss:.6f}")
            print("TARGET_REACHED")
            break

    total_time = time.time() - start_time
    print(f"TOTAL_TIME,{total_time:.3f}")
    print(f"FINAL_STATUS,{val_loss < cfg.TARGET_LOSS}")
if __name__ == "__main__":
    run_training()
