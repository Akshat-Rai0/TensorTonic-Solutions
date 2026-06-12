def remove_stopwords(tokens, stopwords):
    """
    Returns: list[str] - tokens with stopwords removed (preserve order)
    """
    ret = []
    for token in tokens:
        if token not in stopwords:
            ret.append(token)
    return ret 