

def find_max_substring_occurrence(input_string):
    n = len(input_string)
    for i in range(1, n//2 + 1):
        if n % i == 0:
            t = input_string[:i]
            if t * (n // i) == input_string:
                return n // i
    return 1
