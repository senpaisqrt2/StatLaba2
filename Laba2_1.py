# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import scipy.stats as stats
#
# # Чтение данных из файла "Москва_2021.txt"
# file_path = "C:\\Users\\nexti\\Desktop\\Uni\\3 course\\Stat\\Laba2\\Москва_2021.txt"
# with open(file_path, 'r') as file:
#     crime_ages = [int(line.strip()) for line in file]
#
# def calculate_sample_size(sigma, delta, gamma, z_value):
#     return (z_value * sigma / delta) ** 2
#
# # Параметры задачи
# gamma = 0.95
# delta = 3
# z_value = stats.norm.ppf((1 + gamma) / 2)
# sigma = np.std(crime_ages, ddof=0)
#
# # Расчет объема выборки
# n = int(np.ceil(calculate_sample_size(sigma, delta, gamma, z_value)))
# print(f"Рассчитанный объем выборки: {n}")
#
# # Генерация одной выборки
# sample = np.random.choice(crime_ages, size=n, replace=True)
#
# # Расчет выборочного среднего и стандартного отклонения
# sample_mean = np.mean(sample)
# sample_std = np.std(sample, ddof=1)
#
# # Расчет доверительного интервала с использованием t-распределения
# t_value = stats.t.ppf((1 + gamma) / 2, df=n-1)
# margin_of_error = t_value * sample_std / np.sqrt(n)
#
# # Доверительный интервал для математического ожидания
# confidence_interval = (float(sample_mean - margin_of_error), float(sample_mean + margin_of_error))
#
# # Вывод результатов
# print(f"Точечная оценка математического ожидания: {sample_mean:.2f}")
# print(f"Стандартное отклонение выборки: {sample_std:.2f}")
# print(f"Точность (погрешность): {margin_of_error:.2f}")
# print(f"Доверительный интервал: [{confidence_interval[0]:.2f}, {confidence_interval[1]:.2f}]")
#
# # Построение гистограммы выборки
# plt.hist(sample, bins=30, density=True, alpha=0.75, color='g', edgecolor='black')
# plt.title('Гистограмма выборки')
# plt.xlabel('Возраст')
# plt.ylabel('Относительная частота')
# plt.grid(True)
# plt.show()