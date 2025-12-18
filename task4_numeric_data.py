#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import numpy   
import matplotlib.pyplot as plt  
from mpl_toolkits.mplot3d import axes3d, Axes3D
import random 
from sklearn.cluster import BisectingKMeans
from sklearn.metrics import adjusted_rand_score 

NumSamples = 100
'''
Алгоритм подбираем из списка в https://scikit-learn.org/1.5/modules/clustering.html
или реализуем самостоятельно. Применим метод, не использующийся в заданиях:
Bisecting K-Means

Используем два облака точек (вместо трёх в заданиях):
1.  Точки вида (a, b + 0.5, ab), где значения a, b — неза-
висимые случайные числа, равномерно распределён-
ные на отрезке [−1, 1]
2.  Точки вида (a+2, b, 0.5 + a - b), где значения a, b —
независимые случайные числа, нормально распреде-
лённые со средним значением 0 и среднеквадратиче-
ским отклонением 0.5
'''
def generate_samples1(NumSamples):
    # создание первого облака точек
    a = numpy.random.uniform(low=-1, high=1, size=(NumSamples, 1))
    b = numpy.random.uniform(low=-1, high=1, size=(NumSamples, 1))
    v2 = b+0.5
    v3 = a*b
    return numpy.concatenate((a, v2, v3), axis=1)

def generate_samples2(NumSamples):
    # создание второго облака точек
    a = numpy.random.normal(loc=0, scale=0.5, size=(NumSamples, 1))
    b = numpy.random.normal(loc=0, scale=0.5, size=(NumSamples, 1))
    v1 = a+2
    v3 = 0.5 + a - b
    return numpy.concatenate((v1, b, v3), axis=1)

def draw_samples(v1,v2):
    # отрисовка по-разному двух облаков точек
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(v1[:,0], v1[:,1], v1[:,2], label='samples 1')
    ax.scatter(v2[:,0], v2[:,1], v2[:,2], label='samples 2')
    ax.set_ylabel('v1')
    ax.set_ylabel('v2')
    ax.set_zlabel('v3')
    plt.show()

def clustering_bisect_k_mean(samples):
    # обучаем алгоритм кластеризации и сразу рассчитываем результат
    # на обучающих данных
    results = BisectingKMeans(n_clusters=2).fit_predict(samples)
    # кластеризуем данные, выводим ярлыки
    return results

# задаём исходное значение генератору случайных чисел
numpy.random.seed(17)
NumSamples=10
# создаём входные данные
sample1 = generate_samples1(NumSamples)
sample2 = generate_samples2(NumSamples)
# отрисовываем входные данные
draw_samples(sample1, sample2)
# выполняем кластеризацию
full_input = numpy.concatenate((sample1, sample2), axis=0)
cluster_res = clustering_bisect_k_mean(full_input)
# разделяем кластеры
cl0 = full_input[cluster_res == 0,:]
cl1 = full_input[cluster_res == 1,:]
# отрисовываем кластеры
draw_samples(cl0, cl1)
# оцениваем их качество методом Adjusted Rand Index
ari = adjusted_rand_score([0]*NumSamples + [1]*NumSamples, cluster_res)
print(ari)
