import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

# Чтение данных из файла "Москва_2021.txt"
# Здесь мы считываем данные о преступлениях и возрастах преступников из файла
file_path = "C:\\Users\\nexti\\Desktop\\Uni\\3 course\\Stat\\Laba2\\Москва_2021.txt"
with open(file_path, 'r') as file:
    crime_ages = [int(line.strip()) for line in file]

# Функция для расчета объема выборки на основе заданных параметров
# sigma - стандартное отклонение генеральной совокупности
# delta - точность оценки математического ожидания
# gamma - доверительная вероятность (например, 0.95)
# z_value - квантиль нормального распределения для заданной доверительной вероятности
def calculate_sample_size(sigma, delta, gamma, z_value):
    # Формула для расчета объема выборки через нормальное распределение
    return (z_value * sigma / delta) ** 2

# Параметры задачи
gamma = 0.95  # Доверительная вероятность
delta = 3     # Точность оценки математического ожидания (в годах)
z_value = stats.norm.ppf((1 + gamma) / 2)  # Квантиль нормального распределения для доверительной вероятности 0.95
sigma = np.std(crime_ages, ddof=1)  # Стандартное отклонение выборки на основе всех данных о возрастах

# Расчет объема выборки с использованием формулы для нормального распределения
n = int(calculate_sample_size(sigma, delta, gamma, z_value))
print(f"Рассчитанный объем выборки: {n}")

# Генерация 36 выборок и расчет выборочных средних
# Мы создаем 36 случайных выборок с повторениями из исходных данных и находим среднее для каждой выборки
sample_means = [np.mean(np.random.choice(crime_ages, size=n, replace=True)) for _ in range(36)]

# Построение интервального ряда распределения выборочных средних
# Находим минимальное и максимальное значение выборочных средних и округляем их для построения гистограммы
min_sample_mean = np.floor(min(sample_means))  # Округляем вниз минимальное значение
max_sample_mean = np.ceil(max(sample_means))   # Округляем вверх максимальное значение
bins = np.arange(min_sample_mean, max_sample_mean + 1)  # Создаем интервалы (бины) по 1 году

# Построение гистограммы выборочных средних
# Это позволит нам визуализировать распределение выборочных средних
plt.hist(sample_means, bins=bins, density=True, alpha=0.75, color='g', edgecolor='black')
plt.title('Гистограмма выборочных средних')
plt.xlabel('Выборочные средние')
plt.ylabel('Относительная частота')
plt.grid(True)
plt.show()

# Оценка параметров нормального распределения методом моментов
# Среднее и стандартное отклонение (оценки параметров нормального распределения)
mean_est = np.mean(sample_means)  # Оценка среднего
std_est = np.std(sample_means, ddof=1)  # Оценка стандартного отклонения

# Построение кривой Гаусса для аппроксимации гистограммы
x = np.linspace(min_sample_mean, max_sample_mean, 1000)  # Создаем точки для построения кривой Гаусса
pdf = stats.norm.pdf(x, mean_est, std_est)  # Плотность нормального распределения (кривая Гаусса)

# Наложение кривой Гаусса на гистограмму
plt.hist(sample_means, bins=bins, density=True, alpha=0.75, color='g', edgecolor='black', label='Гистограмма')
plt.plot(x, pdf, label='Кривая Гаусса', color='r')  # Кривая Гаусса, аппроксимирующая выборочные средние
plt.title('Аппроксимация гистограммы кривой Гаусса')
plt.xlabel('Выборочные средние')
plt.ylabel('Относительная частота')
plt.legend()
plt.grid(True)
plt.show()

# Доверительный интервал для оценки математического ожидания случайной величины "возраст"
# Здесь используется распределение Стьюдента, так как стандартное отклонение совокупности неизвестно
t_value = stats.t.ppf((1 + gamma) / 2, df=n-1)  # Критическое значение t для доверительной вероятности 0.95
sample_mean = np.mean(sample_means)  # Среднее значение выборочных средних
sample_std = np.std(sample_means, ddof=1)  # Стандартное отклонение выборочных средних

# Вычисление погрешности (запас) для доверительного интервала
margin_of_error = t_value * sample_std / np.sqrt(n)

# Доверительный интервал для математического ожидания
confidence_interval = (float(sample_mean - margin_of_error), float(sample_mean + margin_of_error))
print(f"Доверительный интервал: {confidence_interval}")
