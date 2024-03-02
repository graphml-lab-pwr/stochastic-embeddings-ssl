from functools import partial


def step_fn(num_warmup_steps: int, step: int) -> float:
    if step < num_warmup_steps:
        return float(step) / float(max(1, num_warmup_steps))
    else:
        return 1.0


def linear_warmup_decay(num_warmup_steps: int) -> partial:
    return partial(step_fn, num_warmup_steps)


def compute_warmup(
    num_training_steps: int, num_warmup_steps: int | float
) -> float | int:
    return (
        num_warmup_steps * num_training_steps
        if isinstance(num_warmup_steps, float)
        else num_training_steps
    )


def exclude_bias_and_norm(p):
    return p.ndim == 1
