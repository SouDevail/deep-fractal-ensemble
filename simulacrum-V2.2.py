"""SouDevail. Simulacrum V2.2. Success/Final clean architecture with split RNN bias."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

torch.manual_seed(0)

d_in, d_compact, d_common, d_out = 64, 6, 8, 10
n_macro, k_macro, n_nano, k_nano, tau = 1000, 50, 5, 2, 1.0


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


if __name__ == '__main__':
    m = DFSHE()
    total = sum(p.numel() for p in m.parameters())
    shared = sum(p.numel() for n, p in m.named_parameters()
                 if n in ('compress.weight', 'compress.bias',
                          'macro_router.weight', 'macro_router.bias',
                          'head.weight', 'head.bias'))
    per_me = (total - shared) // n_macro
    print(f'params={total} per_micro={per_me}')
    x = torch.randn(4, d_in)
    y, mask = m(x)
    sparsity = mask.sum().item() * k_nano / (mask.numel() * n_nano) * 100
    print(f'out={tuple(y.shape)} mask={tuple(mask.shape)} sparsity={sparsity:.2f}%')
    m.eval()
    with torch.no_grad():
        y1, _ = m(x)
        y2, _ = m(x)
    print(f'eval_deterministic={torch.allclose(y1, y2)}')
    m.train()
    with torch.no_grad():
        for B in [1, 64]:
            xb = torch.randn(B, d_in)
            t0 = time.time()
            for _ in range(5):
                m(xb)
            dt = (time.time() - t0) / 5 * 1000
            print(f'b={B} time={dt:.1f}ms')
