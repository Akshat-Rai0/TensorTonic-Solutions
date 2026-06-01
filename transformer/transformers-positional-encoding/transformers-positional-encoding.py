import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    pe = np.zeros((seq_length, d_model))

    positions = np.arange(seq_length).reshape(-1, 1)

    div_term = np.exp(
        np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model)
    )

    pe[:, 0::2] = np.sin(positions * div_term)
    pe[:, 1::2] = np.cos(positions * div_term)

    return pe