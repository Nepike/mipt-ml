"""
Практическая работа 4. Кластеризация числовых и текстовых данных
Вариант 1
"""

import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import davies_bouldin_score, mutual_info_score, adjusted_rand_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def generate_points(n_samples=200, point_type=1):
	if point_type == 1:
		# Точки вида (a, b, 0.5 + a + b + ε)
		a = np.random.uniform(-1, 1, n_samples)
		b = np.random.uniform(-1, 1, n_samples)
		epsilon = np.random.uniform(-1, 1, n_samples)
		points = np.column_stack([a, b, 0.5 + a + b + epsilon])
		labels = np.zeros(n_samples)  # метка 0 для типа 1

	elif point_type == 2:
		# Точки вида (a, b+0.5, a*ε)
		a = np.random.uniform(-1, 1, n_samples)
		b = np.random.uniform(-1, 1, n_samples)
		epsilon = np.random.uniform(-1, 1, n_samples)
		points = np.column_stack([a, b + 0.5, a * epsilon])
		labels = np.ones(n_samples)  # метка 1 для типа 2

	return points, labels


if __name__ == '__main__':
	np.random.seed(123)
	points1, labels1 = generate_points(500, point_type=1)
	points2, labels2 = generate_points(500, point_type=2)

	X = np.vstack([points1, points2])
	true_labels = np.hstack([labels1, labels2])

	fig = plt.figure(figsize=(12, 5))
	ax1 = fig.add_subplot(121, projection='3d')
	ax1.scatter(points1[:, 0], points1[:, 1], points1[:, 2], c='blue', alpha=0.6, label='Тип 1', s=30)
	ax1.scatter(points2[:, 0], points2[:, 1], points2[:, 2], c='red', alpha=0.6, label='Тип 2', s=30)
	ax1.set_title('Исходные данные (с метками)')
	ax1.set_xlabel('X')
	ax1.set_ylabel('Y')
	ax1.set_zlabel('Z')
	ax1.legend()

	# KMeans
	kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
	cluster_labels = kmeans.fit_predict(X)

	ax2 = fig.add_subplot(122, projection='3d')
	ax2.scatter(X[:, 0], X[:, 1], X[:, 2], c=cluster_labels, cmap='viridis', alpha=0.6, s=30)
	ax2.scatter(kmeans.cluster_centers_[:, 0],
				kmeans.cluster_centers_[:, 1],
				kmeans.cluster_centers_[:, 2],
				c='red', marker='X', s=200, label='Центроиды')
	ax2.set_title('Результат кластеризации KMeans')
	ax2.set_xlabel('X')
	ax2.set_ylabel('Y')
	ax2.set_zlabel('Z')
	ax2.legend()

	plt.tight_layout()
	plt.show()

	print("Оценка качества кластеризации:")
	db_index = davies_bouldin_score(X, cluster_labels)
	print(f"  Davies-Bouldin Index: {db_index:.4f}")

	mi_score = mutual_info_score(true_labels, cluster_labels)
	print(f"  Mutual Information: {mi_score:.4f}")
	print()

	# ----------------------------------------------------
	# ТЕКСТЫ
	# ----------------------------------------------------

	#positive = negative = ""
	# with open("./texts/positive_reviews.txt") as f:
	# 	positive = f.read()
	# with open("./texts/positive_reviews.txt") as f:
	# 	negative = f.read()

	positive_reviews = [
		"This movie was absolutely fantastic! The acting was superb and the plot kept me engaged throughout.",
		"I loved every minute of this film. The characters were well-developed and the story was compelling.",
		"An excellent piece of cinema. The cinematography was beautiful and the soundtrack was perfect.",
		"One of the best movies I've seen this year. Highly recommended to everyone!",
		"The director did an amazing job. The emotional depth of this film is incredible.",
		"Brilliant performance by the entire cast. The story was both touching and inspiring.",
		"A masterpiece of modern filmmaking. Every scene was carefully crafted and meaningful.",
		"I was completely captivated from beginning to end. Truly a remarkable film.",
		"The visual effects were stunning and the storyline was original and engaging.",
		"This film exceeded all my expectations. A must-watch for any cinema lover!"
	]

	negative_reviews = [
		"This was a terrible movie. The plot made no sense and the acting was awful.",
		"I was very disappointed. The story was boring and the characters were poorly developed.",
		"Waste of time and money. The film dragged on with no clear direction.",
		"The worst movie I've seen in years. I couldn't wait for it to end.",
		"Poorly executed with bad pacing. The dialogue was cringe-worthy.",
		"I regret watching this film. The plot holes were massive and the ending was unsatisfying.",
		"The acting was wooden and the storyline was predictable and unoriginal.",
		"A complete disaster. The director should be ashamed of this production.",
		"I found myself checking my phone multiple times because I was so bored.",
		"This film was a huge letdown. The trailer was misleading and the actual movie was terrible."
	]

	texts = positive_reviews + negative_reviews
	text_labels = [0] * len(positive_reviews) + [1] * len(negative_reviews)

	print(f"Собрано текстов: {len(texts)}")
	print(f"  Положительных отзывов: {len(positive_reviews)}")
	print(f"  Отрицательных отзывов: {len(negative_reviews)}")
	print()

	vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
	X_text = vectorizer.fit_transform(texts)

	kmeans_text = KMeans(n_clusters=2, random_state=42, n_init=10)
	text_clusters_kmeans = kmeans_text.fit_predict(X_text)

	db_text_kmeans = davies_bouldin_score(X_text.toarray(), text_clusters_kmeans)
	mi_text_kmeans = mutual_info_score(text_labels, text_clusters_kmeans)
	print("Оценка качества кластеризации текстов:")
	print(f"  Davies-Bouldin Index: {db_text_kmeans:.4f}")
	print(f"  Mutual Information: {mi_text_kmeans:.4f}")
	print()

	cluster_0_positive = sum(1 for i in range(10) if text_clusters_kmeans[i] == 0)
	cluster_0_negative = sum(1 for i in range(10, 20) if text_clusters_kmeans[i] == 0)
	cluster_1_positive = 10 - cluster_0_positive
	cluster_1_negative = 10 - cluster_0_negative

	print(f"  Распределение по кластерам:")
	print(f"    Кластер 0: {cluster_0_positive} положительных, {cluster_0_negative} отрицательных")
	print(f"    Кластер 1: {cluster_1_positive} положительных, {cluster_1_negative} отрицательных")
	print()
	# ВСЁ ПЛОХО (

	# Попытки улучшить
	improved_positive = [review + " cool like" for review in positive_reviews]
	improved_negative = [review + " bad worst" for review in negative_reviews]
	improved_texts = improved_positive + improved_negative

	X_text_improved = vectorizer.fit_transform(improved_texts)
	kmeans_improved = KMeans(n_clusters=2, random_state=42, n_init=10)
	improved_clusters = kmeans_improved.fit_predict(X_text_improved)

	db_improved = davies_bouldin_score(X_text_improved.toarray(), improved_clusters)
	mi_improved = mutual_info_score(text_labels, improved_clusters)

	print(f"После добавления общих слов:")
	print(f"  Davies-Bouldin Index: {db_improved:.4f} (было {db_text_kmeans:.4f})")
	print(f"  Mutual Information: {mi_improved:.4f} (было {mi_text_kmeans:.4f})")
	print()

	cluster_0_positive = sum(1 for i in range(10) if improved_clusters[i] == 0)
	cluster_0_negative = sum(1 for i in range(10, 20) if improved_clusters[i] == 0)
	cluster_1_positive = 10 - cluster_0_positive
	cluster_1_negative = 10 - cluster_0_negative

	print(f"  Новое распределение по кластерам:")
	print(f"    Кластер 0: {cluster_0_positive} положительных, {cluster_0_negative} отрицательных")
	print(f"    Кластер 1: {cluster_1_positive} положительных, {cluster_1_negative} отрицательных")
	print()








