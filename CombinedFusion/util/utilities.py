import inspect
from torch import nn


def describe(obj) -> str:
    """
    Return a verbose, human-readable string of all public attributes on obj.
    """
    attrs = {}
    for name, value in inspect.getmembers(obj, lambda v: not inspect.isroutine(v)):
        if name.startswith('_'):
            continue
        attrs[name] = value

    body = ', '.join(f"{k}={v!r}" for k, v in attrs.items())
    return f"{obj.__class__.__name__}({body})"


def freeze_model(model: nn.Module) -> nn.Module:
    """
    Freezes all parameters of `model` (no gradients) and switches it to eval mode.
    Returns the same model for convenience (e.g. chaining).
    """
    # 1) Disable gradient computation on all parameters
    for param in model.parameters():
        param.requires_grad = False

    # 2) Switch every sub-module to eval mode
    model.eval()

    return model
