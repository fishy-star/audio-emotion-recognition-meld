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

    # Self loops
    for i in range(num_nodes):

        edges.append([i, i])
        edge_types.append(0)

    # Context window
    for i in range(num_nodes):

        for j in range(
            max(0, i - window),
            min(num_nodes, i + window + 1)
        ):

            if i != j:

                edges.append([i, j])
                edge_types.append(1)

    # Speaker edges
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

        speakers = None
        speaker_ids = None

        if "speaker" in dialogue.columns:

            speakers = dialogue["speaker"].astype(
                str
            ).tolist()

            speaker_map = {}

            speaker_ids = []

            for speaker in speakers:

                if speaker not in speaker_map:

                    speaker_map[speaker] = len(
                        speaker_map
                    )

                speaker_ids.append(
                    speaker_map[speaker]
                )

            speaker_ids = np.array(
                speaker_ids,
                dtype=np.int64
            )

        edge_index, edge_type = build_edge_index(

            num_nodes=len(indices),

            speakers=speakers,

            window=2
        )

        graphs.append(

            {

                "dialogue_id": dialogue_id,

                "x": x,

                "y": y,

                "speaker_ids": speaker_ids,

                "edge_index": edge_index,

                "edge_type": edge_type

            }

        )

    return graphs, encoder