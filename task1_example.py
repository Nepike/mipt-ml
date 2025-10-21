#!/usr/bin/env python3

'''
Пример практической работы 1: регрессия средствами scikit-learn

Задание:
A. Сгенерировать выборку данных, состоящую из элементов (трёхмерных точек) вида
    (x, 0.1 x + y, x^3 + y^3 + epsilon),
где x --- случайное число, равномерно распределённое в диапазоне [-1, 1),
y --- случайное число, нормально распределённое со средним значением 0.2
    и среднеквадратическим отклонением 1.3,
epsilon --- случайное число, нормально распределённое со средним значением 0
    и среднеквадратическим отклонением 0.1

Б. Вывести эту выборку в виде трёхмерного облака точек

В. Провести регрессию на данной выборке методом Elastic Net
    (сочетанием регуляризаций ридж-регрессии и лассо-регрессии)
    средствами библиотеки scikit-learn

Г. Вывести выборку и предсказания на один график
   для визуальной оценки результатов

Д. Оценить ошибку данного метода

'''

# импортируем используемые библиотеки:
import numpy  # базовая библиотека для численных методов
import sklearn.linear_model  # библиотека линейных моделей (в т.ч. линейной регрессии)
import matplotlib.pyplot as plt  # библиотека для вывода данных
from mpl_toolkits.mplot3d import axes3d, Axes3D  # дополнительные библиотеки


# для 3D визуализации

# А. Функция для генерации тестовых данных средствами numpy.random:
# https://numpy.org/doc/stable/reference/random/index.html

def generate_samples(NumSamples=100):
    ''' функция генерации требуемой выборки;
    NumSamples --- её размер '''
    # создаём матрицу случайных действительных чисел для первого параметра (x)
    # размером (NumSamples строк x 1 столбец),
    # равномерно распределённых в диапазоне [-1, 1):
    # https://numpy.org/doc/stable/reference/random/generated/numpy.random.uniform.html
    v1 = numpy.random.uniform(low=-1, high=1, size=(NumSamples, 1))
    # создаём матрицу того же размера для второго параметра:
    # https://numpy.org/doc/stable/reference/random/generated/numpy.random.normal.html
    v2 = v1 + numpy.random.normal(loc=0.2, scale=1.3, size=(NumSamples, 1))
    # создаём матрицу того же размера для правильного ответа:
    epsilon = numpy.random.normal(loc=0, scale=0.1, size=(NumSamples, 1))
    res = numpy.power(v1, 3) + numpy.power(v2, 3) + epsilon
    # комбинируем первый и второй столбцы в одну матрицу, чтобы
    # в каждой её строке были все независимые переменные
    v = numpy.concatenate((v1, v2), axis=1)
    #return (v, res)
    return (v, res.ravel())


# Б. Функция для вывода выборки
def draw_samples(v, res):
    ''' v --- матрица независимых переменных,
    res --- вектор зависимой переменной '''
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(v[:, 0], v[:, 1], res, label='samples')
    ax.set_xlabel('x')
    ax.set_ylabel('0.1x + y')
    ax.set_zlabel('x^3 + y^3 + epsilon')
    plt.show()


# В. Функция для вывода
def calculate_regression_by_standard_method(v, res):
    '''
    рассчитать регрессионную модель встроенными средствами
    библиотеки scikit learn:
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html

    Модель Elastic Net минимизирует
    значение 1 / (2 * n_samples) * ||y - Xw||^2_2
    + alpha * l1_ratio * ||w||_1
    + 0.5 * alpha * (1 - l1_ratio) * ||w||^2_2,
    т.е взвешенную сумму трёх компонентов:
    1) квадратов уклонений предсказаний от результатов (как в обычной линейной регрессии);
    2) L1-нормы вектора параметров (близко по смыслу к лассо-регрессии, хотя и не то же самое);
    3) L2-нормы вектора параметров (как в ридж-регрессии).
    '''
    # создаём объект модели, передавая в него параметры регуляризации
    model = sklearn.linear_model.ElasticNet(alpha=1., l1_ratio=0.5)
    # обучаем модель на выборке встроенным в неё методом fit
    model.fit(v, res)
    # получаем предсказания для выборки в виде вектора
    # того же формата, что и res
    prediction = model.predict(v)
    # возвращаем предсказания
    return prediction


# Г. Функция для вывода выборки вместе с предсказаниями
def draw_samples_and_prediction(v, res, prediction):
    ''' v --- матрица независимых переменных,
    res --- вектор зависимой переменной,
    prediction --- вектор значений этой зависимой переменной,
       предсказанной по v '''
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(v[:, 0], v[:, 1], res, label='samples', marker='o')
    ax.scatter(v[:, 0], v[:, 1], prediction, label='prediction', marker='^')
    ax.set_xlabel('x')
    ax.set_ylabel('0.1x + y')
    ax.set_zlabel('x^3 + y^3 + epsilon')
    plt.show()


# Д. Функция для расчёта ошибки предсказания (среднего отклонения
#    предсказанного значения функции от реального).
def calc_error(res, prediction):
    '''
    res и prediction --- векторы-столбцы одного размера
    '''
    return numpy.linalg.norm(res - prediction) / res.shape[0]


# сгенерируем выборку в виде двух матриц, чтобы предсказываемое значение
# находилось в отдельной переменной

# зададим фиксированное начальное состояние генератора случайных чисел,
# чтобы можно было воспроизвести результаты (т.е. в выборку при каждом
# запуске будут входить одни и те же псевдослучайные числа).
numpy.random.seed(17001)

# создаём выборку
v, res = generate_samples(100)

# выводим её для визуальной оценки
draw_samples(v, res)

# обучаем модель
prediction = calculate_regression_by_standard_method(v, res)

# выводим предсказания
draw_samples(v, prediction)

# выводим одновременно предсказания и исходные данные
draw_samples_and_prediction(v, res, prediction)

# рассчитываем ошибку предсказания
error = calc_error(res, prediction)
print('MSE is {:.3f}'.format(error))