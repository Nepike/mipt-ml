import numpy as np
from scipy.spatial.distance import cdist


class MyGaussKernelSmoothing:
	"""
	Это НЕ аналог KernelRidge из sklearn
	Тут реализовано локальное взвешивание (Nadaraya-Watson)
	"""
	def __init__(self, training_samples, training_answers, sigma=None):
		self.training_samples = training_samples
		self.training_answers = training_answers

		# Вычисляем gamma по аналогии с sklearn (1/(n_features * variance))
		if sigma is None:
			n_features = self.training_samples.shape[1]
			data_variance = np.var(self.training_samples, axis=0).mean()
			self.sigma = 1.0 / (n_features * data_variance) if data_variance > 0 else 1.0
		else:
			self.sigma = sigma

		print(f"Сигма: {self.sigma}")

	def predict(self, test_samples):
		distances = cdist(test_samples, self.training_samples, metric='euclidean')
		weights = np.exp(-distances ** 2 / (2 * self.sigma ** 2))

		numerator = np.dot(weights, self.training_answers)
		denominator = np.sum(weights, axis=1)

		denominator[denominator == 0] = 1e-12

		return numerator / denominator[:, None]
