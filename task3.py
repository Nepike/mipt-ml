#!/usr/bin/env python3
"""
Практическая работа 3. Многослойный персептрон
Вариант 7

Первый класс — точки внутри любого из двух непересекающихся квадратов.
Второй класс — все остальные точки.
"""

import math
import time

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from matplotlib import cm
from sklearn.metrics import precision_score, recall_score
from joblib import Parallel, delayed
from mpl_toolkits.mplot3d import axes3d, Axes3D


def logistic(x, s=10):
	return 1 / (1 + np.exp(-s * x))


def sign(x):
	return (x > 0).astype(float)


def relu(x):
	return np.maximum(0, x)


# ============================================================
# Ручная реализация MLP (без обучения)
# ============================================================

class ManualMLP:
	def __init__(self, coeffs, intercepts, act_func):
		self.coeffs = coeffs
		self.intercepts = intercepts
		self.act_func = act_func

	def predict(self, X):
		curr_val = X
		intermediates = []

		for W, b in zip(self.coeffs, self.intercepts):
			z = curr_val @ W + b
			a = self.act_func(z)
			intermediates.extend([z, a])
			curr_val = a

		return curr_val, intermediates


def draw_all_layers_output(
    net,
    min_x, max_x, min_y, max_y,
    dx, dy,
    figsize=(14, 10)
):
    nx = int(round((max_x - min_x) / dx)) + 1
    ny = int(round((max_y - min_y) / dy)) + 1

    xs = np.linspace(min_x, max_x, nx)
    ys = np.linspace(min_y, max_y, ny)
    vx, vy = np.meshgrid(xs, ys)

    points = np.column_stack([vx.ravel(), vy.ravel()])

    _, interm = net.predict(points)

    num_layers = len(interm) // 2
    max_neurons = max(interm[2 * i].shape[1] for i in range(num_layers))

    fig, axs = plt.subplots(
        nrows=max_neurons,
        ncols=num_layers * 2,
        figsize=figsize,
        squeeze=False
    )

    # Заголовки колонок
    for layer in range(num_layers):
        axs[0, layer * 2].set_title(f'Layer {layer+1}\nlinear', pad=20)
        axs[0, layer * 2 + 1].set_title(f'Layer {layer+1}\nactivated', pad=20)

    for layer in range(num_layers):
        z = interm[2 * layer]
        a = interm[2 * layer + 1]

        for neuron in range(z.shape[1]):
            img_z = np.flipud(z[:, neuron].reshape(ny, nx))
            img_a = np.flipud(a[:, neuron].reshape(ny, nx))

            axs[neuron, layer * 2].imshow(img_z, cmap=cm.coolwarm)
            axs[neuron, layer * 2 + 1].imshow(img_a, cmap=cm.coolwarm)

            # подпись нейрона слева
            axs[neuron, 0].set_ylabel(f'N{neuron+1}', rotation=0, labelpad=25)

        # скрываем неиспользуемые строки
        for neuron in range(z.shape[1], max_neurons):
            axs[neuron, layer * 2].axis('off')
            axs[neuron, layer * 2 + 1].axis('off')

    # убираем оси
    for ax in axs.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


if __name__ == "__main__":
	# Коэффициенты сети, рассчитанные вручную
	layer1_c = np.array([
		[0, 0, 0, 1, 1, 1, 1],
		[1, 1, 1, 0, 0, 0, 0]
	])
	layer1_i = np.array([-1, -2, -5, -1, -2, -3, -6])

	layer2_c = np.array([
		[0, 0, 0, 0, 0, 0, 1],
		[0, 0, 1, 0, 0, 0, 0],
		[0, -1, 0, 0, 1, 0, -1],
		[0, 1, -1, 0, 0, -1, 0],
		[-1, 0, 0, 0, -1, 0, 0],
		[0, -1, 0, -1, 0, 0, 0],
		[1, -1, 0, 1, -1, 0, 0],
		[0, 1, -1, 0, 0, 1, -1],
	]).T
	layer2_i = np.array([-0.5, -0.5, -0.5, -0.5, 0.5, 0.5, -1.5, -1.5])

	layer3_c = np.array([
		[0, 1],
		[0, 1],
		[0, 1],
		[0, 1],
		[0, 1],
		[0, 1],
		[1, 0],
		[1, 0]
	])
	layer3_i = np.array([-0.5, -0.5])

	mlp = ManualMLP(
		coeffs=[layer1_c, layer2_c, layer3_c],
		intercepts=[layer1_i, layer2_i, layer3_i],
		act_func=sign
	)

	draw_all_layers_output(mlp, -1, 7, -1, 7,0.1, 0.1, figsize=(16, 9))


	# Обучим персептрон и сравним результаты

	N = 10000
	xs = np.random.uniform(0, 8, size=(N, 1))
	ys = np.random.uniform(0, 8, size=(N, 1))

	line1 = ys - 1 > 0
	line2 = ys - 2 > 0
	line3 = ys - 5 > 0
	line4 = xs - 1 > 0
	line5 = xs - 2 > 0
	line6 = xs - 3 > 0
	line7 = xs - 6 > 0

	area1 = line1 & ~line2 & line4 & ~line5
	area2 = line2 & ~line3 & line6 & ~line7
	class_ids = 1 * (np.logical_or(area1, area2))

	points = np.hstack((xs, ys))

	points_train, points_test, classes_train, classes_test = train_test_split(points, class_ids,
																			  train_size=0.8,
																			  random_state=17,
																			  shuffle=True)
	# нарисуем обучающую и тестовую выборки
	# fig = plt.figure()
	# ax = fig.add_subplot(projection='3d')
	# ax.scatter(points_train[:, 0], points_train[:, 1], classes_train, c='b', s=5, depthshade=False)
	# ax.scatter(points_test[:, 0], points_test[:, 1], classes_test, c='r', s=5, depthshade=False)
	# ax.set_xlabel('x')
	# ax.set_ylabel('y')
	# ax.set_zlabel('номер класса')
	# plt.show()

	# проверяемые размеры слоёв
	layer1_sizes = [5, 6, 7, 8, 9]
	layer2_sizes = [6, 7, 8, 9, 10]

	precisions = np.zeros((len(layer1_sizes), len(layer2_sizes)))
	recalls = np.zeros((len(layer1_sizes), len(layer2_sizes)))

	# число повторов проверок (т.е. число обучаемых нейросетей каждой конфигурации)
	# NE = 5
	#
	#
	# def train_and_evaluate(l1, l2, e_num):
	# 	mlp = MLPClassifier(hidden_layer_sizes=(l1, l2), activation='logistic', max_iter=2000)
	# 	mlp.fit(points_train, np.ravel(classes_train))
	# 	classes_predicted = mlp.predict(points_test)
	# 	rep = classification_report(classes_test, classes_predicted, output_dict=True, zero_division=0)
	# 	print(f'Network with l1={l1} and l2={l2} processed in experiment {e_num + 1}')
	# 	return rep['1']['precision'], rep['1']['recall']
	#
	#
	# results = Parallel(n_jobs=12)(
	# 	delayed(train_and_evaluate)(l1, l2, e_num)
	# 	for e_num in range(NE)
	# 	for l1 in layer1_sizes
	# 	for l2 in layer2_sizes
	# )
	#
	# results = np.array(results)
	#
	# # разделяем на два массива
	# precisions_all = results[:, 0]
	# recalls_all = results[:, 1]
	#
	# # усредняем по числу повторов NE
	# # для этого сначала reshape к форме (NE, len(layer1_sizes), len(layer2_sizes))
	# precisions = precisions_all.reshape(NE, len(layer1_sizes), len(layer2_sizes)).mean(axis=0)
	# recalls = recalls_all.reshape(NE, len(layer1_sizes), len(layer2_sizes)).mean(axis=0)
	#
	# # выведем тепловые карты данных. Определим границы выводимых значений,
	# # предполагая при этом, что количества нейронов взяты подряд
	# extents = [layer2_sizes[0] - 0.5, layer2_sizes[-1] + 0.5,
	# 		   layer1_sizes[0] - 0.5, layer1_sizes[-1] + 0.5]
	# plt.imshow(precisions, origin='lower', extent=extents)
	# plt.colorbar()
	# plt.title('Precision')
	# plt.xlabel('Размер второго слоя')
	# plt.ylabel('Размер первого слоя')
	# plt.show()
	# plt.imshow(recalls, origin='lower', extent=extents)
	# plt.colorbar()
	# plt.title('Recalls')
	# plt.xlabel('Размер второго слоя')
	# plt.ylabel('Размер первого слоя')
	# plt.show()


	# ЗАДАНИЕ 3
	# N = 20000
	#
	# # 1. Точки внутри куба со стороной 1 и центром (0.5, 0, 0)
	# cube_points = np.random.uniform([0, -0.5, -0.5], [1, 0.5, 0.5], size=(N, 3))
	#
	# # 2. Точки ниже плоскости x - y + z = -3
	# plane_points = np.random.uniform([-2, -2, -2], [2, 2, 2], size=(5 * N, 3))
	# plane_points = plane_points[plane_points[:, 0] - plane_points[:, 1] + plane_points[:, 2] < -3]
	# plane_points = plane_points[:N]
	#
	# # 3. Точки между плоскостями 0.2x + 0.2y + z = 1 и 0.2x + 0.2y + z = -1
	# between_points = np.random.uniform([-2, -2, -2], [2, 2, 2], size=(5 * N, 3))
	# z_values = 0.2 * between_points[:, 0] + 0.2 * between_points[:, 1] + between_points[:, 2]
	# between_points = between_points[(z_values > -1) & (z_values < 1)]
	# between_points = between_points[:N]
	#
	#
	# class0_points = np.vstack([cube_points, plane_points])
	# class1_points = between_points
	#
	# class0_labels = np.zeros(len(class0_points))
	# class1_labels = np.ones(len(class1_points))
	#
	# all_points = np.vstack([class0_points, class1_points])
	# all_labels = np.hstack([class0_labels, class1_labels])
	#
	# # fig = plt.figure(figsize=(16, 9))
	# # ax1 = fig.add_subplot(111, projection='3d')
	# #
	# # ax1.scatter(class0_points[:, 0], class0_points[:, 1], class0_points[:, 2],
	# # 			c='blue', alpha=0.5, s=20, label='Class 0 (cube + below plane)', depthshade=True)
	# #
	# # ax1.scatter(class1_points[:, 0], class1_points[:, 1], class1_points[:, 2],
	# # 			c='red', alpha=0.5, s=20, label='Class 1 (between planes)', depthshade=True)
	# #
	# # ax1.set_xlabel('X')
	# # ax1.set_ylabel('Y')
	# # ax1.set_zlabel('Z')
	# # ax1.set_title('3D Визуализация выборки')
	# # ax1.legend()
	# # ax1.grid(True)
	# #
	# # plt.tight_layout()
	# # plt.show()
	#
	# X_train, X_test, y_train, y_test = train_test_split(
	# 	all_points, all_labels, test_size=0.3, random_state=42, stratify=all_labels
	# )
	#
	# # ОБУЧАЕМ
	# neuron_counts = np.arange(5, 55, 5)  # 5, 10, 15, ..., 50 нейронов
	# n_networks = len(neuron_counts)
	#
	# train_times = []
	# train_precisions = []
	# train_recalls = []
	# test_precisions = []
	# test_recalls = []
	#
	# for i, n_neurons in enumerate(neuron_counts):
	# 	print(f"\nОбучаем сеть {i + 1}/{n_networks}: {n_neurons} нейронов в первом слое")
	#
	# 	mlp = MLPClassifier(
	# 		hidden_layer_sizes=(n_neurons, 10),
	# 		activation='logistic',
	# 		max_iter=2000,
	# 		validation_fraction=0.1
	# 	)
	#
	# 	start_time = time.time()
	# 	mlp.fit(X_train, y_train)
	# 	end_time = time.time()
	#
	# 	train_time = end_time - start_time
	# 	train_times.append(train_time)
	# 	print(f"  Время обучения: {train_time:.2f} сек")
	#
	# 	y_train_pred = mlp.predict(X_train)
	# 	y_test_pred = mlp.predict(X_test)
	#
	# 	train_precision = precision_score(y_train, y_train_pred, zero_division=0)
	# 	train_recall = recall_score(y_train, y_train_pred, zero_division=0)
	#
	# 	test_precision = precision_score(y_test, y_test_pred, zero_division=0)
	# 	test_recall = recall_score(y_test, y_test_pred, zero_division=0)
	#
	# 	train_precisions.append(train_precision)
	# 	train_recalls.append(train_recall)
	# 	test_precisions.append(test_precision)
	# 	test_recalls.append(test_recall)
	#
	# 	print(f"  Train: Precision={train_precision:.3f}, Recall={train_recall:.3f}")
	# 	print(f"  Test:  Precision={test_precision:.3f}, Recall={test_recall:.3f}")
	#
	# fig, axes = plt.subplots(2, 2, figsize=(12, 10))
	#
	# # График 1: Время обучения от числа нейронов
	# axes[0, 0].plot(neuron_counts, train_times, 'bo-', linewidth=2, markersize=8)
	# axes[0, 0].set_xlabel('Число нейронов в первом скрытом слое')
	# axes[0, 0].set_ylabel('Время обучения (сек)')
	# axes[0, 0].set_title('Зависимость времени обучения от числа нейронов')
	# axes[0, 0].grid(True, alpha=0.3)
	# axes[0, 0].set_xticks(neuron_counts)
	#
	# # График 2: Precision для train и test
	# axes[0, 1].plot(neuron_counts, train_precisions, 'go-', linewidth=2, markersize=8, label='Train')
	# axes[0, 1].plot(neuron_counts, test_precisions, 'ro-', linewidth=2, markersize=8, label='Test')
	# axes[0, 1].set_xlabel('Число нейронов в первом скрытом слое')
	# axes[0, 1].set_ylabel('Precision')
	# axes[0, 1].set_title('Precision на обучающей и тестовой выборках')
	# axes[0, 1].legend()
	# axes[0, 1].grid(True, alpha=0.3)
	# axes[0, 1].set_xticks(neuron_counts)
	# axes[0, 1].set_ylim(0, 1.05)
	#
	# # График 3: Recall для train и test
	# axes[1, 0].plot(neuron_counts, train_recalls, 'go-', linewidth=2, markersize=8, label='Train')
	# axes[1, 0].plot(neuron_counts, test_recalls, 'ro-', linewidth=2, markersize=8, label='Test')
	# axes[1, 0].set_xlabel('Число нейронов в первом скрытом слое')
	# axes[1, 0].set_ylabel('Recall')
	# axes[1, 0].set_title('Recall на обучающей и тестовой выборках')
	# axes[1, 0].legend()
	# axes[1, 0].grid(True, alpha=0.3)
	# axes[1, 0].set_xticks(neuron_counts)
	# axes[1, 0].set_ylim(0, 1.05)
	#
	# # График 4: Разница между train и test (признак переобучения)
	# train_avg = np.array(train_precisions) + np.array(train_recalls)
	# test_avg = np.array(test_precisions) + np.array(test_recalls)
	# difference = train_avg - test_avg
	#
	# axes[1, 1].bar(neuron_counts, difference, width=3, alpha=0.7)
	# axes[1, 1].set_xlabel('Число нейронов в первом скрытом слое')
	# axes[1, 1].set_ylabel('Разница (train - test)')
	# axes[1, 1].set_title('Признак переобучения')
	# axes[1, 1].grid(True, alpha=0.3)
	# axes[1, 1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
	# axes[1, 1].set_xticks(neuron_counts)
	#
	# plt.tight_layout()
	# plt.show()
