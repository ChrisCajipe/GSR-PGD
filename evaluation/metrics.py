def calculate_binary_metrics(predictions, labels):

    tp = tn = fp = fn = 0


    for pred, label in zip(predictions, labels):

        if pred == 1 and label == 1:
            tp += 1

        elif pred == 0 and label == 0:
            tn += 1

        elif pred == 1 and label == 0:
            fp += 1

        elif pred == 0 and label == 1:
            fn += 1


    total = len(labels)


    accuracy = (
        (tp + tn) / total
        if total > 0
        else 0
    )


    tpr = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0
    )


    tnr = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0
    )


    return {
        "accuracy": accuracy,
        "tpr": tpr,
        "tnr": tnr,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn
    }