#!/usr/bin/env python3

import math

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d, Axes3D


# В задании 2 работы 3 требуется определить число нейронов, требуемое
# для решения задачи классификации точек двумерного пространства.
# Чтобы проверить выводы, можно реализовать маленький персептрон
# с явно задаваемыми синаптическими коэффициентами,
# чтобы получить доступ ко всем выходам его нейронов. Сделаем это
# для задачи классификации, рассмотренной на лекции по нейросетям
# на слайдах 229--243

def logistic(x, s=10):
	# логистическая функция активации
	return 1 / (1 + np.exp(-s * x))


def sign(x):
	# пороговая функция активации
	return (x > 0) * 1.


def relu(x):
	# ReLU-функция активации
	return x if x > 0 else 0.


class ManualMLP:
	# чтобы не лезть в детали реализации многослойного персептрона
	# в scikit-learn, сделаем свой маленький класс для него (без метода обучения,
	# т.е. только с ручным заданием коэффициентов)
	def __init__(self, coeffs, intercepts, act_func):
		# создать сеть по списку коэффициентов и функции активации
		self.coeffs = coeffs  # синаптические коэффициенты при входах
		self.intercepts = intercepts  # коэффициенты - постоянные (смещения нуля)
		self.act_func = act_func  # функция активации

	def predict(self, X):
		# получить выходы всех слоёв, а не только последнего
		curr_val = X
		results = []
		for i in range(len(self.coeffs)):
			# для каждого слоя провести линейное преобразование предыдущего
			curr_val = np.matmul(curr_val, self.coeffs[i]) + self.intercepts[i]
			# поместить его выход в результаты
			results.append(curr_val)
			# применить к этому же выходу нелинейное преобразование
			curr_val2 = self.act_func(curr_val)
			# поместить его тоже в результаты
			curr_val = curr_val2
			results.append(curr_val2)
		# выдать из функции и выход, и список результатов каждого слоя
		return curr_val, results


# создадим массивы коэффициентов, которые были в явном виде рассчитаны
# на лекции для задачи классификации
layer1_c = [[1, -0.5, 1, -2],
			[1, 1, 1, 1]]
layer1_i = [-5, -0.5, -8, 1]

layer2_c = [[0, -1, 1, 0, 0],
			[1, 1, 0, 0, -1],
			[1, 0, -1, 0, 0],
			[-1, -1, 0, 1, 0]]
layer2_i = [-1.5, -0.5, -0.5, -0.5, 0.5]

layer3_c = [[1., 0],
			[1, 0],
			[0, 1],
			[0, 1],
			[0, 1]]
layer3_i = [-0.5, -0.5]

# создадим сеть с этими коэффициентами и пороговыми функциями активации всех слоёв
mlp = ManualMLP([np.array(layer1_c), np.array(layer2_c), np.array(layer3_c)],
				[np.array(layer1_i), np.array(layer2_i), np.array(layer3_i)],
				sign)


# обучать сеть не надо, потому что коэффициенты для неё мы рассчитали вручную;
# можно просто вывести её результаты для рассматриваемой области данных

def draw_all_layers_output(net, min_x, max_x, min_y, max_y, dx, dy):
	# функция вывода результатов работы всех нейронов сети net;
	# принимает она два числа -- первое принимает значения
	# от min_x до max_x с шагом dx, второе --- от min_y до max_y с шагом dy
	# Вычислим размер входных данных
	nx = round((max_x - min_x) / dx) + 1
	ny = round((max_y - min_y) / dy) + 1
	# составим из них решётку --- пару матриц vx,vy, в которых
	# сочетаются все возможные пары x и y
	px = np.linspace(min_x, max_x, nx)
	py = np.linspace(min_x, max_x, ny)
	vx, vy = np.meshgrid(px, py)
	test_set = np.vstack([np.ravel(vx), np.ravel(vy)]).T
	# предскажем классы всех элементов решётки на нейросети
	preds, interm = net.predict(test_set)

	# выведем эти классы
	fig, axs = plt.subplots(nrows=5, ncols=6)
	# размеры слоёв сети
	szs = [interm[i].shape[1] for i in range(len(interm))]

	for i, sz in enumerate(szs):
		for j in range(sz):
			# для каждого используемого нейрона вывести выход в виде изображения
			reshaped_data = np.reshape(interm[i][:, j], (nx, ny))
			dt = np.flip(reshaped_data, 0)
			axs[j, i].imshow(dt, cmap=cm.coolwarm)
			axs[j, i].set_xticks([])
			axs[j, i].set_yticks([])
			# неиспользуемыми оставить пустыми
			for j in range(sz, max(szs)):
				axs[j, i].set_xticks([])
				axs[j, i].set_yticks([])
	plt.show()


# выведем выходы всех нейронов сети mlp в диапазонах x и y [-3,10]
# с шагом x и y 0.1. Слои выводятся слева направо; для каждого нейрона
# выводится сначала сумма входов, а потом результат действия функции активации
# на эту сумму.

draw_all_layers_output(mlp, -3, 10, -3, 10, 0.1, 0.1)

# По выведенной картине можно понять, удалось ли решить задачу классификации
# на построенной вручную нейросети. Можно использовать эту сеть для проверки
# правильности решения задания 2.

# Теперь попытаемся обучить сеть с тем числом нейронов, которое было получено
# на прошлом этапе, и с близкими к нему, чтобы понять, насколько оправдаются
# предсказания относительно требуемого числа нейронов. Входов и выходов у сети
# всегда будет по два, а число нейронов скрытых слоёв будем варьировать.

# Сгенерируем выборку данных для обучения: возьмём множество случайных,
# равномерно распределённых точек из квадрата [-3..10,-3..10], проверим их
# позиции относительно линий, разделяющих классы, и выведем номера классов

N = 10000
# x и y равномерно покрывают плоскость
xs = np.random.uniform(-3, 10, size=(N, 1))
ys = np.random.uniform(-3, 10, size=(N, 1))
# выделим классы по той же логике, что в лекции
line1 = xs + ys - 5 > 0
line2 = -0.5 * xs + ys - 0.5 > 0
line3 = xs + ys - 8 > 0
line4 = -2 * xs + ys + 1 > 0
# выпуклые области получим сочетанием линейных разбиений, как в лекции
area1 = np.logical_and(np.logical_and(line2, line3), np.logical_not(line4))
area2 = np.logical_and(np.logical_and(np.logical_not(line1), np.logical_not(line4)), line2)
# так как классов всего два, достаточно выделить один из них
class_ids = 1 * (np.logical_or(area1, area2))

# выборка будет состоять из точек points и номеров их классов class_ids
points = np.hstack((xs, ys))

# разделим единую начальную выборку на обучающую и тестовую.
# Классификатор будет обучаться только на первой из них, а качество
# работы будет оцениваться по второй. Это нужно для распознавания переобучения:
# если классификатор намного лучше сработает на обучающей выборке,
# чем на тестовой, он слишком приспособился к конкретным примерам
# и потерял способность обобщать данные.

points_train, points_test, classes_train, classes_test = train_test_split(points, class_ids,
																		  train_size=0.8,
																		  random_state=17,
																		  shuffle=True)

# нарисуем обучающую и тестовую выборки
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(points_train[:, 0], points_train[:, 1], classes_train, 'b')
ax.scatter(points_test[:, 0], points_test[:, 1], classes_test, 'r')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('номер класса')
plt.show()

# проверяемые размеры слоёв
layer1_sizes = [2, 3, 4, 5]
layer2_sizes = [2, 3, 4, 5, 6]

# массивы для результатов
precisions = np.zeros((len(layer1_sizes), len(layer2_sizes)))
recalls = np.zeros((len(layer1_sizes), len(layer2_sizes)))

# число повторов проверок (т.е. число обучаемых нейросетей каждой конфигурации)
NE = 3

for e_num in range(NE):
	for i, layer1_size in enumerate(layer1_sizes):
		for j, layer2_size in enumerate(layer2_sizes):
			# сделаем сеть с числом нейронов скрытых слоёв,
			# равным (layer1_size, layer2_size). Так как
			# функция активация будет отличаться от sign, результат вполне может
			# быть другим. Можно проверить его, заменив 'logistic' на 'relu' или 'tanh'
			mlp = MLPClassifier(hidden_layer_sizes=(layer1_size, layer2_size),
								activation='logistic')
			# обучаем классификатор на обучающей выборке
			mlp.fit(points_train, np.ravel(classes_train))
			# предсказываем классы для тестовой выборки
			classes_predicted = mlp.predict(points_test)
			# вырабатываем оценку качества
			rep = classification_report(classes_test, classes_predicted, output_dict=True)
			# выберем из результатов precision и recall для класса 1
			precisions[i, j] += rep['1']['precision']
			recalls[i, j] += rep['1']['recall']
			print(f'Network with l1={layer1_size} and l2={layer2_size} processed in experiment {e_num + 1}')

# усредняем собранные данные
precisions /= NE
recalls /= NE
# выводим их в явном виде на всякий случай
print(precisions)
print(recalls)

# выведем тепловые карты данных. Определим границы выводимых значений,
# предполагая при этом, что количества нейронов взяты подряд
extents = [layer2_sizes[0] - 0.5, layer2_sizes[-1] + 0.5,
		   layer1_sizes[0] - 0.5, layer1_sizes[-1] + 0.5]
plt.imshow(precisions, origin='lower', extent=extents)
plt.colorbar()
plt.title('Precision')
plt.xlabel('Первый слой')
plt.ylabel('Второй слой')
plt.show()
plt.imshow(recalls, origin='lower', extent=extents)
plt.colorbar()
plt.title('Recalls')
plt.xlabel('Первый слой')
plt.ylabel('Второй слой')
plt.show()
