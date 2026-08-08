from .weighted_ce import weighted_cross_entropy
from .lovasz import lovasz_softmax

def combined_loss(logits, target, class_weights):
    wce = weighted_cross_entropy(logits, target, class_weights)
    lovasz = lovasz_softmax(logits, target, classes="present")
    return wce + lovasz, {"weighted_ce": wce, "lovasz": lovasz}
