import torch
from hydra.utils import instantiate


def poly_lr(epoch: int, max_epochs: int, power: float = 0.9) -> float:
    # LambdaLR expects a multiplicative factor
    # clamp to avoid negative for safety if epoch > max_epochs
    t = max(0.0, 1.0 - float(epoch) / float(max_epochs))
    return t ** power



def instantiate_scheduler(sched_node, optimizer: torch.optim.Optimizer):
    """
    DDP/Hydra-safe scheduler instantiation.
    Special-cases SequentialLR so nested schedulers get the real optimizer object.
    """
    target = getattr(sched_node, "_target_", None) or sched_node.get("_target_", None)
    if target is None:
        raise ValueError("cfg.optim.scheduler must contain a `_target_` key.")

    # ---- Special case: SequentialLR (nested schedulers need optimizer) ----
    if target == "torch.optim.lr_scheduler.SequentialLR":
        scheds_cfg = sched_node.get("schedulers", None)
        milestones = sched_node.get("milestones", None)
        if scheds_cfg is None or milestones is None:
            raise ValueError("SequentialLR requires `schedulers` and `milestones` keys.")

        sub_schedulers = [instantiate(sub_cfg, optimizer=optimizer) for sub_cfg in scheds_cfg]

        return torch.optim.lr_scheduler.SequentialLR(
            optimizer=optimizer,
            schedulers=sub_schedulers,
            milestones=list(milestones),
        )

    # ---- Default: instantiate scheduler with optimizer ----
    return instantiate(sched_node, optimizer=optimizer)
