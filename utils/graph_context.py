import numpy as np
from sklearn.preprocessing import LabelEncoder

def build_edge_index(num_nodes, speakers=None, window=2):
    edges = set()

    for i in range(num_nodes):
        edges.add((i, i))

    for i in range(num_nodes):
        for j in range(max(0, i - window), min(num_nodes, i + window + 1)):
            if i != j:
                edges.add((i, j))
                edges.add((j, i))

    if speakers is not None:
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j and speakers[i] == speakers[j]:
                    edges.add((i, j))
                    edges.add((j, i))

    edge_index = np.array(list(edges), dtype=np.int64).T
    return edge_index

def build_graphs(embeddings, metadata, encoder=None):
    metadata = metadata.copy()

    if encoder is None:
        encoder = LabelEncoder()
        encoder.fit(metadata["emotion"])

    metadata["label"] = encoder.transform(metadata["emotion"])

    graphs = []

    for dialogue_id, dialogue in metadata.groupby("dialogue_id", sort=False):
        dialogue = dialogue.sort_values("utterance_id")
        indices = dialogue.index.tolist()

        x = embeddings[indices].astype(np.float32)
        y = dialogue["label"].values.astype(np.int64)

        speakers = dialogue["speaker"].astype(str).tolist() if "speaker" in dialogue.columns else None
        edge_index = build_edge_index(len(indices), speakers=speakers, window=2)

        graphs.append({
            "dialogue_id": dialogue_id,
            "x": x,
            "y": y,
            "edge_index": edge_index
        })

    return graphs, encoder