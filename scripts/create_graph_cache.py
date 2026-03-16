import json
import pickle
import numpy as np
from pathlib import Path

print("Creating pre-processed graph cache...")

# Load graphs
with open("data/processed/complex_features/complex_graphs.json", "r") as f:
    graphs = json.load(f)

print(f"Processing {len(graphs)} graphs...")

# Pre-process all graphs
processed_graphs = {}
for pdb_id, graph in graphs.items():
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    
    # Pre-compute node features as numpy array
    node_features = []
    for node in nodes:
        if isinstance(node, dict):
            # Use just basic features, not the full 1280-dim saprot
            # We only extract structural features here
            feat = [
                node.get("bfactor", 0),
                node.get("resi", 0),
            ]
            # Add ca_coord if available
            ca = node.get("ca_coord")
            if ca is None:
                ca = [0, 0, 0]
            feat.extend(ca)
        else:
            feat = [0] * 5
        node_features.append(feat)
    
    # Pre-compute edge index
    edge_list = []
    for edge in edges:
        if isinstance(edge, dict):
            edge_list.append([edge.get("source", 0), edge.get("target", 0)])
        elif isinstance(edge, (list, tuple)) and len(edge) >= 2:
            edge_list.append([edge[0], edge[1]])
    
    if not edge_list:
        n = max(len(nodes), 2)
        src = list(range(n - 1)) + list(range(1, n))
        dst = list(range(1, n)) + list(range(n - 1))
        edge_index = np.array([src, dst], dtype=np.int64)
    else:
        edge_index = np.array(edge_list, dtype=np.int64).T
    
    processed_graphs[pdb_id] = {
        "node_features": np.array(node_features, dtype=np.float32),
        "edge_index": edge_index,
        "n_nodes": len(nodes)
    }

# Save processed graphs
cache_path = Path("data/processed/graphs_cache.pkl")
with open(cache_path, "wb") as f:
    pickle.dump(processed_graphs, f)

print(f"✓ Saved processed graphs to {cache_path}")
print(f"  Cache size: {cache_path.stat().st_size / 1024 / 1024:.1f} MB")
