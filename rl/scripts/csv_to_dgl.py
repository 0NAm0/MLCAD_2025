# scripts/csv_to_dgl.py
import os
os.environ["DGL_DISABLE_GRAPHBOLT"] = "1"
import argparse
import json
import pandas as pd


from rl.utils.graph_builder import build_dgl_from_csv  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", required=True, help="Path to nodes.csv")
    ap.add_argument("--edges", required=True, help="Path to edges.csv")
    ap.add_argument("--outdir", default="artifacts/graph", help="Where to save outputs")
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    g, meta = build_dgl_from_csv(args.nodes, args.edges)
    print(g)
    print(meta)

    # degree and Top-K
    indeg = g.in_degrees().numpy()
    outdeg = g.out_degrees().numpy()
    k = args.topk
    top_out_idx = outdeg.argsort()[-k:][::-1]
    top_in_idx = indeg.argsort()[-k:][::-1]

    # save node_id ↔ idx mapping
    nodes_df = pd.read_csv(args.nodes, low_memory=False)
    node_id_col = meta["node_id_col"]
    mapping_df = pd.DataFrame(
        {"node_idx": range(len(nodes_df)), "node_id": nodes_df[node_id_col].astype(str).tolist()}
    )
    mapping_df.to_csv(os.path.join(args.outdir, "node_id_mapping.csv"), index=False)

    # save Top-K
    pd.DataFrame(
        {
            "node_idx": top_out_idx,
            "node_id": mapping_df.loc[top_out_idx, "node_id"].tolist(),
            "out_degree": outdeg[top_out_idx],
        }
    ).to_csv(os.path.join(args.outdir, "top_out_degree.csv"), index=False)

    pd.DataFrame(
        {
            "node_idx": top_in_idx,
            "node_id": mapping_df.loc[top_in_idx, "node_id"].tolist(),
            "in_degree": indeg[top_in_idx],
        }
    ).to_csv(os.path.join(args.outdir, "top_in_degree.csv"), index=False)

    with open(os.path.join(args.outdir, "graph_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[OK] Saved to {args.outdir}")


if __name__ == "__main__":
    main()
