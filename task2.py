"""
Практическая работа 2. Простейшие методы классификации
Вариант 14

Задание:
	1. Сгенерировать выборку данных:
		7. Случайные точки трехмерного пространства, распределённые по поверхности нижней полусферы с уравнением:
		x^2 + y^2 + (z + 0.5)^2 = 1, z ≤ 0 (способ распределения произвольный)

		8. Случайные точки трёхмерного пространства, лежащие внутри полушария:
		x^2 + y^2 + z^2 = 1, z ≥ 0; способ распределения произвольный

		10. Точки трёхмерного пространства вида (x, y, exp(x−y) + ε),
		где x и y — нормально распределённые случайные числа со средним 1 и среднеквадратическим отклонением 3,
		ε — равномерно распределённые случайные числа из диапазона [−0.1, 0.1]

	В обучающую выборку включаются три указанные группы точек трехмерного пространства;
	каждую из классификаций следует провести три раза:
		1. включив в один класс первую группу точек, в другой — вторую и третью;
		2. включив в один класс первую и вторую группы точек, в другой — третью;
		3. включив каждую группу точек в свой отдельный класс.

	2. Обработать каждую из выборок указанными методами и сравнить качество полученных моделей
	LogisticRegression LinearDiscriminantAnalysis GaussianMixture
	(precision, recall, accuracy).

	3. Сгенерировать новую выборку данных такого же размера по шаблону 1.1.
	После этого сравнить точность всех моделей на новой выборке и первой из прежних.

	4. Провести регрессию на основе гауссова процесса. Для этого оставить в одном
	из классов входных данных первоначальной выборки только две координаты (убрать первую
	или вторую из трёх) и, воспользовавшись готовой реализацией из sklearn (gaussian_process.
	GaussianProcessRegressor), провести обработку, а затем вывести на график исходные данные
	и предсказания, полученные по модели. Использовать для оценки ковариации одно из ядер,
	перечисленных в модуле sklearn.gaussian_process.kernels.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.mixture import GaussianMixture

from sklearn.metrics import precision_score, recall_score, accuracy_score
from scipy.stats import mode

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel


def generate_type7(n):
	points = []
	while len(points) < n:
		x, y, z = np.random.uniform(-1, 1, 3)  # Случайные точки в кубе [-1, 1]
		if abs(x**2 + y**2 + (z + 0.5)**2 - 1) < 0.002 and z <= 0:
			points.append((x, y, z))
	return np.array(points)


def generate_type8(n):
	points = []
	while len(points) < n:
		x, y, z = np.random.uniform(-1, 1, 3)
		if x**2 + y**2 + z**2 <= 1 and z >= 0:
			points.append((x, y, z))
	return np.array(points)


def generate_type10(n):
	x = np.random.normal(1, 3, n)
	y = np.random.normal(1, 3, n)
	eps = np.random.uniform(-0.1, 0.1, n)
	z = (np.exp(x - y) + eps)*1
	return np.column_stack((x, y, z))


def make_datasets(data7, data8, data10):
	X1 = np.vstack([data7, data8, data10])
	y1 = np.hstack([
		np.zeros(len(data7)),
		np.ones(len(data8)),
		np.ones(len(data10))
	])

	X2 = np.vstack([data7, data8, data10])
	y2 = np.hstack([
		np.zeros(len(data7)),
		np.zeros(len(data8)),
		np.ones(len(data10))
	])

	X3 = np.vstack([data7, data8, data10])
	y3 = np.hstack([
		np.zeros(len(data7)),
		np.ones(len(data8)),
		np.full(len(data10), 2)
	])

	return (X1, y1), (X2, y2), (X3, y3)


def plot_dataset(X, y, title="Визуализация данных", y_real=None, focus_range: float = 2):
	fig = plt.figure(figsize=(10, 8))
	ax = fig.add_subplot(111, projection='3d')

	unique_classes = np.unique(y)
	base_colors = ["#4ecdc4", "#556270", "#ffa600", "#6a4c93", "#1e90ff"]
	colors = base_colors[:len(unique_classes)]

	for idx, cls in enumerate(unique_classes):
		mask = y == cls
		ax.scatter(
			X[mask, 0], X[mask, 1], X[mask, 2],
			color=colors[idx % len(colors)],
			s=25, alpha=0.85,
			label=f"Класс {int(cls)}"
		)

	if y_real is not None:
		wrong_mask = y != y_real
		if np.any(wrong_mask):
			ax.scatter(
				X[wrong_mask, 0],
				X[wrong_mask, 1],
				X[wrong_mask, 2],
				color="#ff0000",
				s=60,
				alpha=1.0,
				edgecolor='k',
				linewidth=0.3,
				label="Ошибки"
			)

	ax.set_title(title, fontsize=14, pad=12)
	ax.set_xlabel("x")
	ax.set_ylabel("y")
	ax.set_zlabel("z")
	ax.legend()
	ax.grid(True, linestyle=":", alpha=0.5)

	# Фокус на центре
	ax.set_xlim(-focus_range, focus_range)
	ax.set_ylim(-focus_range, focus_range)
	ax.set_zlim(-focus_range, focus_range)
	ax.set_box_aspect([1, 1, 1])

	mng = plt.get_current_fig_manager()
	try:
		mng.window.state('zoomed')
	except Exception:
		try:
			mng.full_screen_toggle()
		except Exception:
			pass

	plt.tight_layout()
	plt.show()


def train_all_models(X_train, y_train):

	models = {}

	logreg = LogisticRegression(max_iter=1000)
	logreg.fit(X_train, y_train)
	models["LR"] = logreg

	lda = LinearDiscriminantAnalysis()
	lda.fit(X_train, y_train)
	models["LDA"] = lda

	# обучается как unsupervised, потом сопоставим классы
	gmm = GaussianMixture(n_components=len(np.unique(y_train)), covariance_type='full', random_state=42)
	gmm.fit(X_train)
	models["GMM"] = gmm

	return models


def evaluate_models(models, X_test, y_test):
	results = {}

	for name, model in models.items():
		if name == "GMM":
			y_pred = model.predict(X_test)

			# Для многоклассовой классификации используем оптимальное сопоставление кластеров и классов
			from scipy.optimize import linear_sum_assignment

			# Создаем матрицу совпадений между кластерами и реальными классами
			n_clusters = model.n_components
			n_classes = len(np.unique(y_test))

			contingency_matrix = np.zeros((n_clusters, n_classes))
			for i in range(n_clusters):
				for j in range(n_classes):
					contingency_matrix[i, j] = np.sum((y_pred == i) & (y_test == j))

			# Находим оптимальное сопоставление с помощью венгерского алгоритма
			row_ind, col_ind = linear_sum_assignment(-contingency_matrix)

			# Создаем mapping
			mapping = np.zeros(n_clusters, dtype=int)
			mapping[row_ind] = col_ind

			y_pred = mapping[y_pred]
		else:
			y_pred = model.predict(X_test)

		results[name] = {
			"precision": precision_score(y_test, y_pred, average='macro', zero_division=0),
			"recall": recall_score(y_test, y_pred, average='macro', zero_division=0),
			"accuracy": accuracy_score(y_test, y_pred),
			"y_pred": y_pred,
			"precision_per_class": precision_score(y_test, y_pred, average=None, zero_division=0),
			"recall_per_class": recall_score(y_test, y_pred, average=None, zero_division=0)
		}

	return results


def print_metrics(metrics, model_name):
	print(f"  {model_name}:\n"
		  f"    accuracy={metrics['accuracy']:.3f}\n"
		  f"    precision (macro)={metrics['precision']:.3f}\n"
		  f"    recall (macro)={metrics['recall']:.3f}")
	if 'precision_per_class' in metrics:
		print(f"    precision per class={np.round(metrics['precision_per_class'], 3)}\n"
			  f"    recall per class={np.round(metrics['recall_per_class'], 3)}")


def compare_old_new(results_old, results_new, scheme_name):
	print(f"\n=== Сравнение моделей для {scheme_name}: старая vs новая выборка ===")
	for model_name in results_old.keys():
		old_metrics = results_old[model_name]
		new_metrics = results_new[model_name]

		print(f"\nМодель: {model_name}")
		print("  Старая выборка:")
		for i, (p, r) in enumerate(zip(old_metrics['precision_per_class'], old_metrics['recall_per_class'])):
			print(f"    Класс {i}: precision={p:.3f}, recall={r:.3f}")
		print(f"    accuracy={old_metrics['accuracy']:.3f}")

		print("  Новая выборка:")
		for i, (p, r) in enumerate(zip(new_metrics['precision_per_class'], new_metrics['recall_per_class'])):
			print(f"    Класс {i}: precision={p:.3f}, recall={r:.3f}")
		print(f"    accuracy={new_metrics['accuracy']:.3f}")


def gaussian_process_regression_plot(data, input_idx=(0, 1), output_idx=2, title="Gaussian Process Regression"):
	X_gp = data[:, input_idx]
	y_gp = data[:, output_idx]

	# Определяем ядро: RBF + константа
	kernel = ConstantKernel(1.0, (1e-2, 1e2)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
	gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=0.01)

	gpr.fit(X_gp, y_gp)
	y_pred, sigma = gpr.predict(X_gp, return_std=True)

	# Визуализация
	fig = plt.figure(figsize=(10, 7))
	ax = fig.add_subplot(111, projection='3d')

	# исходные точки
	ax.scatter(X_gp[:, 0], X_gp[:, 1], y_gp, color='blue', s=40, alpha=0.6, label='Исходные данные')

	# предсказания
	ax.scatter(X_gp[:, 0], X_gp[:, 1], y_pred, color='red', s=40, alpha=0.6, label='Предсказания GPR')

	ax.set_xlabel(f'X[{input_idx[0]}]')
	ax.set_ylabel(f'X[{input_idx[1]}]')
	ax.set_zlabel(f'Y[{output_idx}]')
	ax.set_title(title)
	ax.legend()
	plt.show()

	return gpr, y_pred, sigma


if __name__ == '__main__':
	np.random.seed(271025)

	N_samples = 300

	data7 = generate_type7(N_samples)
	data8 = generate_type8(N_samples)
	data10 = generate_type10(N_samples)

	(train1_X, train1_y), (train2_X, train2_y), (train3_X, train3_y) = make_datasets(data7, data8, data10)

	# plot_dataset(train1_X, train1_y, "Схема 1: (7) vs (8 + 10)", focus_range=2)
	# plot_dataset(train2_X, train2_y, "Схема 2: (7 + 8) vs (10)", focus_range=2)
	# plot_dataset(train3_X, train3_y, "Схема 3: (7) vs (8) vs (10)", focus_range=2)

	models1 = train_all_models(train1_X, train1_y)
	models2 = train_all_models(train2_X, train2_y)
	models3 = train_all_models(train3_X, train3_y)

	results1 = evaluate_models(models1, train1_X, train1_y)
	results2 = evaluate_models(models2, train2_X, train2_y)
	results3 = evaluate_models(models3, train3_X, train3_y)

	for i, (res, name, train_X, train_y) in enumerate([(results1, "Схема 1", train1_X, train1_y),
													   (results2, "Схема 2", train2_X, train2_y),
													   (results3, "Схема 3", train3_X, train3_y)], start=1):
		print(f"\n{name}")
		for model_name, metrics in res.items():
			print_metrics(metrics, model_name)

			# plot_dataset(
			# 	train_X,
			# 	metrics["y_pred"],
			# 	title=f"{name} — {model_name}",
			# 	y_real=train_y,
			# 	focus_range=2
			# )

	data7_new = generate_type7(N_samples)
	data8_new = generate_type8(N_samples)
	data10_new = generate_type10(N_samples)

	(test1_X, test1_y), (test2_X, test2_y), (test3_X, test3_y) = make_datasets(data7_new, data8_new, data10_new)

	results1_new = evaluate_models(models1, test1_X, test1_y)
	results2_new = evaluate_models(models2, test2_X, test2_y)
	results3_new = evaluate_models(models3, test3_X, test3_y)

	for i, (res, name, test_X, test_y) in enumerate([(results1_new, "Схема 1", test1_X, test1_y),
													   (results2_new, "Схема 2", test2_X, test2_y),
													   (results3_new, "Схема 3", test3_X, test3_y)], start=1):
		print(f"\n{name} (НОВАЯ ВЫБОРКА)")
		for model_name, metrics in res.items():
			print_metrics(metrics, model_name)

			# plot_dataset(
			# 	test_X,
			# 	metrics["y_pred"],
			# 	title=f"{name} — {model_name} (НОВАЯ ВЫБОРКА)",
			# 	y_real=test_y,
			# 	focus_range=2
			# )

	# compare_old_new(results1, results1_new, "Схема 1")
	# compare_old_new(results2, results2_new, "Схема 2")
	compare_old_new(results3, results3_new, "Схема 3")

	gpr_model, y_pred, sigma = gaussian_process_regression_plot(data10, input_idx=(0, 1), output_idx=2)

	print(sigma)










