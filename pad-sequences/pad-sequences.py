import numpy as np
def pad_sequences(seqs, pad_value=0, max_len=None):

    if max_len is None:
        max_len = max(len(seq) for seq in seqs)

    result = []

    for seq in seqs:
        seq = seq[:max_len]  # truncate if longer
        seq = seq + [pad_value] * (max_len - len(seq))
        result.append(seq)

    return np.array(result)
        
