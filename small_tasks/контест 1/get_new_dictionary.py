

def get_new_dictionary(input_dict_name, output_dict_name):
    f1 = open(input_dict_name, 'r')
    f2 = open(output_dict_name, 'w')
    n1 = int(f1.readline())
    input_dict = dict()
    for i in range(n1):
        input_string = f1.readline()
        input_string = input_string.rstrip()
        input_list = input_string.split(' - ')
        input_list[1] = input_list[1].split(', ')
        input_dict[input_list[0]] = input_list[1]
    output_dict = dict()
    for key in input_dict.keys():
        for value in input_dict[key]:
            if value in output_dict:
                output_dict[value].append(key)
            else:
                output_dict[value] = key.split()
    f2.write(str(len(output_dict)))
    f2.write('\n')
    for key in sorted(output_dict.keys()):
        f2.write(key)
        f2.write(' - ')
        flag = False
        for value in sorted(output_dict[key]):
            if flag:
                f2.write(', ')
            f2.write(value)
            flag = True
        f2.write('\n')
    f1.close()
    f2.close()
