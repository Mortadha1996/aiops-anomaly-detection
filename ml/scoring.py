def weighted_score(f1, recall, roc_auc):
    return 0.4*f1 + 0.35*recall + 0.25*roc_auc

def choose_best(results):
    return max(results, key=lambda x: x["score"])
