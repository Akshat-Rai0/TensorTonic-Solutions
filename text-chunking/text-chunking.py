def text_chunking(tokens, chunk_size, overlap):
    step = chunk_size - overlap
    result = []

    for i in range(0, len(tokens), step):
        chunk = tokens[i:i + chunk_size]

        if len(chunk) < chunk_size and i != 0:
            break

        result.append(chunk)

    return result