# rl/scripts/train_sage_embed.py
# Minimal self-supervised training to produce node embeddings from CSV graph.

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import dgl
import dgl.nn as dglnn


def build_graph_from_csv(nodes_csv: str, edges_csv: str):
    """
    Build a homogeneous DGLGraph from CSVs.
    nodes.csv must contain a column 'id' (original node ids).
    edges.csv must contain columns 'src' and 'tar' (original ids).
    Returns:
        g: DGLGraph with N nodes and E edges
        id_map: pandas.DataFrame with columns ['node_idx','node_id']
    """
    nodes = pd.read_csv(nodes_csv)
    edges = pd.read_csv(edges_csv)

    assert "id" in nodes.columns, "nodes.csv must have column 'id'"
    assert {"src", "tar"}.issubset(edges.columns), "edges.csv must have 'src' and 'tar'"

    # Map original ids -> contiguous [0..N-1]
    node_ids = nodes["id"].astype(str).tolist()
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

    # Translate edge endpoints to integer indices
    src = edges["src"].astype(str).map(id_to_idx).values
    dst = edges["tar"].astype(str).map(id_to_idx).values

    # Build the graph (directed)
    g = dgl.graph((src, dst), num_nodes=len(node_ids))
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)  # optional: add self loops for SAGE stability

    id_map = pd.DataFrame({"node_idx": range(len(node_ids)), "node_id": node_ids})
    return g, id_map


class SAGE(nn.Module):
    """Two-layer GraphSAGE encoder producing node embeddings."""

    def __init__(self, in_feats: int, hid_feats: int, out_feats: int):
        super().__init__()
        self.conv1 = dglnn.SAGEConv(in_feats, hid_feats, aggregator_type="mean")
        self.conv2 = dglnn.SAGEConv(hid_feats, out_feats, aggregator_type="mean")

    def forward(self, g, x):
        h = self.conv1(g, x)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h  # (N, out_feats)


class DotPredictor(nn.Module):
    """Edge scorer using dot product of node embeddings."""

    def forward(self, h, edges):
        # edges: tuple (u, v) with shape (E,), (E,)
        u, v = edges
        score = (h[u] * h[v]).sum(dim=1)  # logits
        return score


def negative_sample(num_nodes: int, num_edges: int, device: torch.device):
    """Uniform negative edges (may include few collisions; fine for self-supervised pretraining)."""
    u = torch.randint(0, num_nodes, (num_edges,), device=device)
    v = torch.randint(0, num_nodes, (num_edges,), device=device)
    return u, v


def train(g, embed_dim: int, hidden_dim: int, out_dim: int, epochs: int, lr: float, outdir: Path):
    device = torch.device("cpu")  # CPU-only as requested
    g = g.to(device)
    N = g.num_nodes()

    # Use a trainable embedding table as initial node features (no external features yet)
    node_feat = nn.Embedding(N, embed_dim)
    nn.init.xavier_uniform_(node_feat.weight)

    encoder = SAGE(in_feats=embed_dim, hid_feats=hidden_dim, out_feats=out_dim)
    predictor = DotPredictor()

    params = list(encoder.parameters()) + list(node_feat.parameters())
    opt = torch.optim.Adam(params, lr=lr)

    u_pos, v_pos = g.edges()  # all positive edges (with self-loops)
    u_pos, v_pos = u_pos.to(device), v_pos.to(device)

    for epoch in range(1, epochs + 1):
        encoder.train()
        opt.zero_grad()

        x = node_feat.weight  # (N, embed_dim)
        h = encoder(g, x)     # (N, out_dim)

        # Positive scores
        pos_score = predictor(h, (u_pos, v_pos))

        # Negative edges: sample the same number as positives
        u_neg, v_neg = negative_sample(N, len(u_pos), device)
        neg_score = predictor(h, (u_neg, v_neg))

        # Binary cross-entropy with logits
        labels = torch.cat([
            torch.ones_like(pos_score, device=device),
            torch.zeros_like(neg_score, device=device)
        ])
        logits = torch.cat([pos_score, neg_score])
        loss = F.binary_cross_entropy_with_logits(logits, labels)

        loss.backward()
        opt.step()

        with torch.no_grad():
            # Simple AUC proxy: logit separation (higher is better)
            auc_proxy = (pos_score.mean() - neg_score.mean()).item()

        print(f"[Epoch {epoch:03d}] loss={loss.item():.4f}  sep={auc_proxy:.4f}")

    # Final embeddings
    encoder.eval()
    with torch.no_grad():
        x = node_feat.weight
        h = encoder(g, x).cpu()  # (N, out_dim)

    outdir.mkdir(parents=True, exist_ok=True)
    torch.save({"embeddings": h}, outdir / "embeddings.pt")
    print(f"[OK] Saved node embeddings to: {outdir / 'embeddings.pt'}")
    return h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", type=str, required=True, help="Path to nodes.csv (must contain 'id')")
    ap.add_argument("--edges", type=str, required=True, help="Path to edges.csv (must contain 'src','tar')")
    ap.add_argument("--outdir", type=str, default="artifacts/embeddings", help="Output directory for artifacts")
    ap.add_argument("--embed-dim", type=int, default=64)
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--out-dim", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    g, id_map = build_graph_from_csv(args.nodes, args.edges)
    outdir = Path(args.outdir)

    # Train encoder and get embeddings
    h = train(
        g,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        out_dim=args.out_dim,
        epochs=args.epochs,
        lr=args.lr,
        outdir=outdir,
    )

    # Save a CSV with embeddings joined to original node ids for traceability
    emb_df = pd.DataFrame(h.numpy())
    emb_df.insert(0, "node_idx", range(len(emb_df)))
    emb_df = emb_df.merge(id_map, on="node_idx", how="left")
    emb_df.to_csv(outdir / "node_embeddings.csv", index=False)
    print(f"[OK] Saved: {outdir / 'node_embeddings.csv'}")


if __name__ == "__main__":
    main()
