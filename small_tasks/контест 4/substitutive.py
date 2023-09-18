from functools import wraps
import inspect


def substitutive(func):
    substitutive._state = []

    @wraps(func)
    def wrapper(*args):
        for x in args:
            substitutive._state.append(x)
        try:
            func(*substitutive._state)
        except Exception as e:
            if len(inspect.getfullargspec(func)[0]) < len(substitutive._state):
                raise TypeError
            return wrapper
        for _ in args:
            substitutive._state.pop(-1)
    return wrapper
