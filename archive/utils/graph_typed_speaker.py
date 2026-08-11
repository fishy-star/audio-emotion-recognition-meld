import numpy as np
from sklearn.preprocessing import LabelEncoder


# Edge types:
# 0 = self loop
# 1 = context window
# 2 = same speaker
# 3 = different speaker


def build_edge_index(
    num_nodes,
    speakers=None,
    window=2
):

    edges = []
    edge_types = []

    for i in range(num_nodes):

        edges.append([i, i])
        edge_types.append(0)


    for i in range(num_nodes):

        for j in range(
            max(0, i-window),
            min(num_nodes, i+window+1)
        ):

            if i != j:

                edges.append([i, j])
                edge_types.append(1)



    if speakers is not None:

        for i in range(num_nodes):

            for j in range(i):

                if speakers[i] == speakers[j]:

                    edges.append([i, j])
                    edge_types.append(2)

                    edges.append([j, i])
                    edge_types.append(2)

                else:

                    edges.append([i, j])
                    edge_types.append(3)

                    edges.append([j, i])
                    edge_types.append(3)



    edge_index = np.array(
        edges,
        dtype=np.int64
    ).T


    edge_type = np.array(
        edge_types,
        dtype=np.int64
    )


    return edge_index, edge_type



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


    # speaker encoder
    speaker_encoder = LabelEncoder()

    speaker_encoder.fit(
        metadata["speaker"].astype(str)
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


        # store speaker ids
        speaker_ids = speaker_encoder.transform(
            dialogue["speaker"].astype(str)
        )


        y = dialogue["label"].values.astype(
            np.int64
        )


        speakers = dialogue["speaker"].astype(
            str
        ).tolist()


        edge_index, edge_type = build_edge_index(
            len(indices),
            speakers,
            window=2
        )


        graphs.append(
            {
                "dialogue_id": dialogue_id,
                "x": x,
                "speaker_ids": speaker_ids,
                "y": y,
                "edge_index": edge_index,
                "edge_type": edge_type
            }
        )


    return graphs, encoder, len(speaker_encoder.classes_)