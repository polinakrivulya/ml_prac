from functools import wraps


def check_arguments(*input_arg):
    def the_real_decorator(function):
        @wraps(function)
        def wrapper(*args):
            if len(input_arg) > len(args):
                raise TypeError
            j = 0
            for i in input_arg:
                if not isinstance(args[j], i):
                    raise TypeError
                j += 1
            result = function(*args)
            return result

        return wrapper

    return the_real_decorator

@substitutive
def f(x, y, z):
    print(x, y, z)

try:
    f(1, 2, 3)
except Exception as e:
    print(e)