import math


def linear_schedule(initial_lr: float):
    """
    Linear learning rate decay function.
    :param initial_lr: Initial learning rate.
    """
    def lr_schedule(progress_remaining: float):
        """
        Decays linearly: progress_remaining goes from 1 (start) to 0 (end).
        """
        return initial_lr * progress_remaining
    return lr_schedule


def exponential_schedule(initial_lr: float, decay_rate: float = 0.99):
    """
    Exponential learning rate decay function.
    :param initial_lr: Initial learning rate.
    :param decay_rate: Decay factor per remaining progress (default 0.99).
    """
    def lr_schedule(progress_remaining: float):
        """
        Decays exponentially: learning rate decreases faster as training progresses.
        """
        return initial_lr * (decay_rate ** (1 - progress_remaining))
    return lr_schedule


def cosine_annealing_schedule(initial_lr: float, min_lr: float = 1e-6):
    """
    Cosine annealing learning rate decay function.
    :param initial_lr: Initial learning rate.
    :param min_lr: Minimum learning rate value.
    """
    def lr_schedule(progress_remaining: float):
        """
        Decays learning rate following a cosine curve.
        """
        return min_lr + (initial_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * (1 - progress_remaining)))
    return lr_schedule
