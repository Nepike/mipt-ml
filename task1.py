"""
Практическая работа 1. Простейшие методы регрессии
Вариант 5

Задание:
	1. Сгенерировать выборку данных:
	Точки двумерного пространства вида (x, x^2),
	где значения x нормально распределены со средним 3 и среднеквадратическим отклонением 1

	2. Обработать выборку методами двумя методами линейной регрессии библиотеки scikit-learn
	(MSE и ядерное сглаживание с ядром в форме гауссианы)
	Сравнить точности полученных регрессионных моделей

	3. Реализовать метод ядерного сглаживания самостоятельно, сравнить его результаты на выборке
	с готовым методом библиотеки scikit-learn.

	4. Создать новую выборку данных такого же размера и сравнить точность моделей
	на старой и новой выборках. Сделать вывод о сохранении точностных характеристик каждой моделей.
"""

import numpy
import sklearn.linear_model
import sklearn.kernel_ridge
import matplotlib.pyplot as plt
from my import MyGaussKernelSmoothing


def generate_samples(samples_num: int):
	arg = numpy.random.normal(loc=3, scale=1, size=(samples_num, 1))
	res = arg**2
	return arg, res


def draw_samples(arg, res):
	plt.figure(figsize=(10, 6))
	plt.scatter(arg, res, alpha=0.5, c='blue', s=10, label='Samples')

	plt.title(f'Samples')
	plt.xlabel('x ~ N(3, 1)')
	plt.ylabel('y = x^2')
	plt.grid(alpha=0.3)
	plt.legend()
	plt.show()


def calculate_regression_by_standard_methods(arg, res):
	model2 = sklearn.kernel_ridge.KernelRidge(alpha=1, kernel='rbf')
	model2.fit(arg, res)
	prediction2 = model2.predict(arg)

	#arg = numpy.column_stack([arg, arg**3, ])
	model1 = sklearn.linear_model.LinearRegression()
	model1.fit(arg, res)
	prediction1 = model1.predict(arg)

	return prediction1, prediction2


def draw_samples_and_predictions(arg, res, prediction1, prediction2):
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

	ax1.scatter(arg, res, alpha=0.5, c='blue', s=10, label='Samples')
	ax1.scatter(arg, prediction1, c='red', s=10, label='LinearRegression')
	ax1.set_title('LinearRegression')
	ax1.set_xlabel('x ~ N(3, 1)')
	ax1.set_ylabel('y = x^2')
	ax1.grid(alpha=0.3)
	ax1.legend()

	ax2.scatter(arg, res, alpha=0.5, c='blue', s=10, label='Samples')
	ax2.scatter(arg, prediction2, c='red', s=10, label='KernelRidge')
	ax2.set_title('KernelRidge')
	ax2.set_xlabel('x ~ N(3, 1)')
	ax2.set_ylabel('y = x^2')
	ax2.grid(alpha=0.3)
	ax2.legend()

	plt.tight_layout()
	plt.show()


def calculate_error(res, prediction):
	return numpy.linalg.norm(res - prediction) / res.shape[0]


def calculate_regression_by_my_and_standard_methods(arg, res):
	model1 = MyGaussKernelSmoothing(arg, res, sigma=None)
	prediction1 = model1.predict(arg)

	model2 = sklearn.kernel_ridge.KernelRidge(alpha=1, kernel='rbf')
	model2.fit(arg, res)
	prediction2 = model2.predict(arg)

	return prediction1, prediction2


if __name__ == '__main__':
	numpy.random.seed(191025)

	arg, res = generate_samples(100)

	#draw_samples(arg, res)

	prediction1, prediction2 = calculate_regression_by_standard_methods(arg, res)

	print(f"LinearRegression error: {calculate_error(res, prediction1) : .3f}")
	print(f"KernelRidge error: {calculate_error(res, prediction2) : .3f}")

	draw_samples_and_predictions(arg, res, prediction1, prediction2)

	prediction1, prediction2 = calculate_regression_by_my_and_standard_methods(arg, res)

	print(f"MyGaussKernelSmoothing error: {calculate_error(res, prediction1) : .3f}")
	print(f"KernelRidge error: {calculate_error(res, prediction2) : .3f}")

	draw_samples_and_predictions(arg, res, prediction1, prediction2)
