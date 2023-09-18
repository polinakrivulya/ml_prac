'''
Можно добавить:

    from sys import *
setrecursionlimit(100000)

В таком случае строки 13-19 (цикл) можно убрать
'''


def find_path_sums(tup):
    def main_func(tree, summ):
        while (tree[1] and not tree[2]) or (tree[2] and not tree[1]):
            if tree[1] and not tree[2]:
                summ += tree[0]
                tree = tree[1]
            if tree[2] and not tree[1]:
                summ += tree[0]
                tree = tree[2]
        if tree[1]:
            main_func(tree[1], tree[0] + summ)
        if tree[2]:
            main_func(tree[2], tree[0] + summ)
        if not tree[1] and not tree[2]:
            print(tree[0] + summ)
    main_func(tup, 0)
