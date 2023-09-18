

def check_first_sentence_is_second(first_sentence, second_sentence):
    first_list = first_sentence.split(' ')
    second_list = second_sentence.split(' ')
    for x in second_list:
        if not x:
            continue
        b = False
        for y in first_list:
            if x == y:
                b = True
                first_list.remove(y)
                break
        if not b:
            return False
    return True
