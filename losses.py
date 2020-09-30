class MaskedMSELoss:
    def __init__(self, pad_value=-1000, weight=1):
        self.pad_value = pad_value
        self.weight = weight

    def __call__(self, prediction, labels):
        loss_mat = (prediction - labels) ** 2
        mask = labels != self.pad_value
        loss = loss_mat[mask].sum()
        return self.weight * loss
