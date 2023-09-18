'''
Видела решения похожей задачи:
https://ru.stackoverflow.com/questions/584595/Сделать-глубоковложенный-список-плоским-без-ветвления-и-циклов
'''


def linearize(iterat):
    def linstr(iterat1):
        return (list(iterat1))
    try:
        if (len(iterat) == 0):
            return []
        if (type(iterat) == str):
            return linstr(iterat)
        first, *second = iterat
        if second == []:
            return linearize(first)
        else:
            return linearize(first) + linearize(second)
    except TypeError:
        return [iterat]
