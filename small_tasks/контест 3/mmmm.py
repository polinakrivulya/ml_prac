import numpy as np
from random import randint
import time
import matplotlib
import matplotlib.pyplot as plt


def get_max_before_zero1(x):
    if not any(x[0: -1] == 0):
        return None
    else:
        y = []
        for i in range(1, len(x)):
            if (x[i - 1] == 0):
                y.append(x[i])
        maximum = y[0]
        for i in range(1, len(y)):
            if y[i] > maximum:
                maximum = y[i]
        return maximum
def get_max_before_zero2(x):
    if not any(x[0: -1] == 0):
        return None
    else:
        lst = []
        for i in range(1, len(x)):
            if (x[i - 1] == 0):
                lst.append(i)
        y = x[lst]
        return np.max(y)
def get_max_before_zero3(x):
    if not any(x[0: -1] == 0):
        return None
    else:
        y = x[np.where(x[0: -1] == 0)[0] + 1]
        return np.max(y)
y = []
time1 = []
time2 = []
time3 = []
i = 10
while (i != 100000000):
    h = np.random.randint(0, 100, i)
    y.append(i)
    start_time = time.time()
    get_max_before_zero1(h)
    time1.append(time.time() - start_time)
    start_time = time.time()
    get_max_before_zero2(h)
    time2.append(time.time() - start_time)
    start_time = time.time()
    get_max_before_zero3(h)
    time3.append(time.time() - start_time)
    i *= 100
fig, ax = plt.subplots()
plt.title('Зависимость времени выполнения от числа данных', color = 'r', fontweight = "bold")
plt.xlabel('Время, секунд', color = 'r')
plt.ylabel('Число данных', color = 'r')
ax.plot(time1, y, color = 'darkblue', linewidth = 3, label = 'Полностью не векторизованная')
ax.plot(time2, y, color = 'darkgreen', linewidth = 3, label = 'Частично векторизованная')
ax.plot(time3, y, color = 'y', linewidth = 3, label = 'Полностью векторисованная')
fig.set_figwidth(12)
fig.set_figheight(7)
ax.grid()
ax.legend(fontsize = 8,
          ncol = 1,    #  количество столбцов
          shadow = False,
          title = 'Реализация',    #  заголовок
          title_fontsize = '10'    #  размер шрифта заголовка
         )
plt.show()