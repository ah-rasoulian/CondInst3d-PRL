def poly_lr(epoch: int, max_epochs: int, power: float = 0.9) -> float:
    # LambdaLR expects a multiplicative factor
    # clamp to avoid negative for safety if epoch > max_epochs
    t = max(0.0, 1.0 - float(epoch) / float(max_epochs))
    return t ** power
