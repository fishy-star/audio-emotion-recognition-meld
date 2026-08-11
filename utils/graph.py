import numpy as np
from sklearn.preprocessing import LabelEncoder


def build_edge_index(num_nodes):

    edges = []

    for i in range(num_nodes):

        edges.append(
            [i, i]
        )

    for i in range(num_nodes - 1):

        edges.append(
            [i, i + 1]
        )

        edges.append(
            [i + 1, i]
        )

    return np.array(
        edges,
        dtype=np.int64
    ).T



def build_graphs(
    embeddings,
    metadata,
    encoder=None
):

    metadata = metadata.copy()


    if encoder is None:

        encoder = LabelEncoder()

        encoder.fit(
            metadata["emotion"]
        )


    metadata["label"] = encoder.transform(
        metadata["emotion"]
    )


    graphs = []


    for dialogue_id, dialogue in metadata.groupby(
        "dialogue_id",
        sort=False
    ):

        dialogue = dialogue.sort_values(
            "utterance_id"
        )


        indices = dialogue.index.tolist()


        x = embeddings[
            indices
        ].astype(
            np.float32
        )


        y = dialogue["label"].values.astype(
            np.int64
        )


        edge_index = build_edge_index(
            len(indices)
        )


        graphs.append(

            {

                "dialogue_id": dialogue_id,

                "x": x,

                "y": y,

                "edge_index": edge_index

            }

        )


    return graphs, encoder