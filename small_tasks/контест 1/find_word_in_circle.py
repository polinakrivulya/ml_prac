

def find_word_in_circle(circle, word):
    len_circle = len(circle)
    if len_circle == 0:
        return -1
    len_word = len(word)
    help_string = (len_word // len_circle + 2) * circle
    index = help_string.find(word)
    if index != -1:
        return index, 1
    help_string = help_string[::-1]
    index = help_string.find(word)
    if index != -1:
        index = len_circle - help_string.find(word) - 1
        return index, -1
    return -1
