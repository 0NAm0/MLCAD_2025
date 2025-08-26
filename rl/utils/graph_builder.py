# rl/utils/graph_builder.py
from __future__ import annotations
import os

os.environ["DGL_DISABLE_GRAPHBOLT"] = "1"

import pandas as pd
import torch
import dgl



def _infer_columns(nodes_df: pd.DataFrame, edges_df: pd.DataFrame):
    """Heuristically infer node id column and edge src/dst columns."""
    # node id
    node_id = None
    for c in nodes_df.columns:
        if c.lower() in ["id", "node_id", "nid", "name", "uid", "index"]:
            node_id = c
            break
    if node_id is None:
        first = nodes_df.columns[0]
        if nodes_df[first].is_unique:
            node_id = first

    # src/dst
    src, dst = None, None
    for c in edges_df.columns:
        lc = c.lower()
        if lc in ["src", "source", "u", "from", "src_id", "start", "head"] and src is None:
            src = c
        if lc in ["dst", "target", "v", "to", "dst_id", "end", "tail", "tar"] and dst is None:
            dst = c

    if src is None or dst is None:
        cols = list(edges_df.columns)
        if len(cols) >= 2:
            src, dst = cols[0], cols[1]
    return node_id, src, dst


def build_dgl_from_csv(nodes_csv: str, edges_csv: str):
    """
    Build a DGLGraph from CSVs. Returns (graph, meta_dict).
    meta_dict includes: node_id_col, src_col, dst_col, num_feat_cols, num_nodes, num_edges
    """
    # Force string columns to avoid dtype warnings; keep the rest as inferred by pandas.
    nodes_df = pd.read_csv(nodes_csv, low_memory=False, dtype={"libcell_name": "string"})
    edges_df = pd.read_csv(edges_csv, low_memory=False)

    node_id, src, dst = _infer_columns(nodes_df, edges_df)
    if node_id is None or src is None or dst is None:
        raise ValueError(f"Column infer failed: node_id={node_id}, src={src}, dst={dst}")

    # map to contiguous ids
    id2idx = {str(nid): i for i, nid in enumerate(nodes_df[node_id].astype(str).tolist())}
    src_idx = edges_df[src].astype(str).map(id2idx).fillna(-1).astype(int)
    dst_idx = edges_df[dst].astype(str).map(id2idx).fillna(-1).astype(int)
    valid = (src_idx >= 0) & (dst_idx >= 0)

    g = dgl.graph(
        (torch.tensor(src_idx[valid].values), torch.tensor(dst_idx[valid].values)),
        num_nodes=len(nodes_df),
    )

    # attach numeric node features if any
    num_cols = [
        c for c in nodes_df.columns if c != node_id and pd.api.types.is_numeric_dtype(nodes_df[c])
    ]
    if num_cols:
        g.ndata["feat"] = torch.tensor(nodes_df[num_cols].values, dtype=torch.float32)
    else:
        g.ndata["feat"] = torch.ones((g.num_nodes(), 1), dtype=torch.float32)

    meta = {
        "node_id_col": node_id,
        "src_col": src,
        "dst_col": dst,
        "num_feat_cols": num_cols,
        "num_nodes": g.num_nodes(),
        "num_edges": g.num_edges(),
    }
    return g, meta
