import numpy as np


def serialize(obj):
    """
    Convert an object to a serializable object. Used for JSON serialization in the hyperparameter tuning script.
    :param obj: result dictionary to serialize
    :return: serialized object
    """

    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, dict):
        return {key: serialize(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [serialize(item) for item in obj]
    return obj
