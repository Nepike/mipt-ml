import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def filter_classes(dataset, class1=0, class2=1):
	targets = np.array(dataset.targets)
	indices = np.where((targets == class1) | (targets == class2))[0]
	return Subset(dataset, indices)


class Autoencoder(nn.Module):
	"""
	- 2 слоя в кодировщике
	- 2 слоя в декодировщике
	- Минимальное количество признаков: 30
	"""

	def __init__(self, latent_dim=30):
		super().__init__()
		input_size = 32 * 32 * 3

		self.encoder = nn.Sequential(
			nn.Linear(input_size, 256),
			nn.ReLU(),
			nn.Linear(256, latent_dim),
			nn.ReLU()
		)

		self.decoder = nn.Sequential(
			nn.Linear(latent_dim, 256),
			nn.ReLU(),
			nn.Linear(256, input_size),
			nn.Tanh()
		)

	def forward(self, x):
		"""
		Args:
			x: Входное изображение в виде тензора [batch_size, channels, height, width]

		Returns:
			reconstructed: Восстановленное изображение
			latent: Скрытое представление (латентный вектор)
		"""

		batch_size = x.size(0)

		x_flat = x.view(batch_size, -1)

		latent = self.encoder(x_flat)
		reconstructed_flat = self.decoder(latent)

		# Преобразуем обратно в 4D формат [batch, channels, height, width]
		reconstructed = reconstructed_flat.view(batch_size, 3, 32, 32)

		return reconstructed, latent


def train_autoencoder(model, train_loader, test_loader, epochs=20, lr=0.001, device=torch.device('cpu')):
	model.to(device)
	optimizer = optim.Adam(model.parameters(), lr=lr)
	criterion = nn.MSELoss()

	train_losses = []
	test_losses = []

	for epoch in range(epochs):
		model.train()
		train_loss = 0.0

		for batch_idx, (data, _) in enumerate(train_loader):
			# data: батч изображений [32, 3, 32, 32]
			data = data.to(device)
			optimizer.zero_grad()
			reconstructed, _ = model(data)
			loss = criterion(reconstructed, data)
			# (backpropagation)
			loss.backward()
			optimizer.step()

			train_loss += loss.item()

		avg_train_loss = train_loss / len(train_loader)
		train_losses.append(avg_train_loss)

		# === ТЕСТОВАЯ ФАЗА ===
		model.eval()
		test_loss = 0.0

		with torch.no_grad():
			for data, _ in test_loader:
				data = data.to(device)
				reconstructed, _ = model(data)
				loss = criterion(reconstructed, data)
				test_loss += loss.item()

		avg_test_loss = test_loss / len(test_loader)
		test_losses.append(avg_test_loss)

		# Выводим статистику каждую эпоху
		print(f"Эпоха [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}")

	return train_losses, test_losses


def visualize_results(model, data_loader, device=torch.device('cpu'), num_images=5):
	model.eval()

	data_iter = iter(data_loader)
	images, _ = next(data_iter)

	images = images[:num_images].to(device)

	with torch.no_grad():
		reconstructed, _ = model(images)

	images = images.cpu()
	reconstructed = reconstructed.cpu()

	images = images * 0.5 + 0.5
	reconstructed = reconstructed * 0.5 + 0.5


	fig, axes = plt.subplots(2, num_images, figsize=(15, 6))

	for i in range(num_images):
		ax = axes[0, i]
		img = images[i].permute(1, 2, 0).numpy()
		ax.imshow(img)
		ax.set_title(f"Исходное {i + 1}")
		ax.axis('off')

		ax = axes[1, i]
		recon_img = reconstructed[i].permute(1, 2, 0).numpy()
		ax.imshow(recon_img)
		ax.set_title(f"Восстановленное {i + 1}")
		ax.axis('off')

	plt.suptitle("Сравнение исходных и восстановленных изображений")
	plt.tight_layout()
	plt.show()


def interpolate_images(model, img1, img2, num_steps=5, device=torch.device('cpu')):
	model.eval()

	img1 = img1.to(device)
	img2 = img2.to(device)

	interpolated_images = []

	with torch.no_grad():
		_, latent1 = model(img1)
		_, latent2 = model(img2)

		for i in range(num_steps + 1):
			alpha = i / num_steps  # 0 → 1

			latent = (1 - alpha) * latent1 + alpha * latent2

			decoded_flat = model.decoder(latent)
			decoded = decoded_flat.view(1, 3, 32, 32)

			decoded = decoded * 0.5 + 0.5

			interpolated_images.append(decoded.cpu())

	return interpolated_images


if __name__ == '__main__':
	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])

	train_dataset = datasets.CIFAR10(
		root='./data',
		train=True,
		download=True,
		transform=transform
	)

	test_dataset = datasets.CIFAR10(
		root='./data',
		train=False,
		download=True,
		transform=transform
	)

	train_dataset = filter_classes(train_dataset, 0, 1)
	test_dataset = filter_classes(test_dataset, 0, 1)

	train_loader = DataLoader(
		train_dataset,
		batch_size=256,
		shuffle=True,
		num_workers=4,
		pin_memory=True,
		persistent_workers=True
	)

	test_loader = DataLoader(
		test_dataset,
		batch_size=256,
		shuffle=False,
		num_workers=4,
		pin_memory=True,
		persistent_workers=True
	)

	# model_30 = Autoencoder(latent_dim=30)
	# train_losses_30, test_losses_30 = train_autoencoder(model_30, train_loader, test_loader, epochs=20)
	# visualize_results(model_30, test_loader)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	models = {}
	test_results = {}
	latent_dims = [30, 60, 120]
	for dim in latent_dims:
		print("\n" + "=" * 50)
		print(f"ОБУЧЕНИЕ МОДЕЛИ С latent_dim={dim}")
		print("=" * 50)

		model = Autoencoder(latent_dim=dim)
		# state_dict = torch.load(
		# 	f'./models/autoencoder_dim_{dim}.pth',
		# 	map_location=device
		# )
		# model.load_state_dict(state_dict)

		_, test_losses = train_autoencoder(model, train_loader, test_loader, epochs=25, device=device)

		models[dim] = model
		test_results[dim] = test_losses[-1]  # потери последней эпохи

	print("\nСравнение MSE для разных размеров латентного пространства:")
	for dim, mse in test_results.items():
		print(f"latent_dim={dim}: MSE = {mse:.4f}")

	dims = list(test_results.keys())
	mses = list(test_results.values())

	plt.figure(figsize=(8, 5))
	plt.plot(dims, mses, 'bo-', linewidth=2, markersize=8)
	plt.xlabel('Размер латентного пространства', fontsize=12)
	plt.ylabel('MSE', fontsize=12)
	plt.title('Зависимость качества восстановления от размера латентного пространства', fontsize=14)
	plt.grid(True, alpha=0.3)
	plt.xticks(dims)

	for i, (dim, mse) in enumerate(zip(dims, mses)):
		plt.text(dim, mse + 0.0002, f'{mse:.4f}', ha='center', va='bottom')

	plt.tight_layout()
	plt.show()

	visualize_results(models[120], test_loader, device=device)

	for dim, model in models.items():
		torch.save(model.state_dict(), f'./models/autoencoder_dim_{dim}.pth')


	# ИНТЕРПОЛЯЦИЯ
	test_iter = iter(test_loader)
	test_images, _ = next(test_iter)

	img1 = test_images[0:1]
	img2 = test_images[1:2]

	interpolated = interpolate_images(
		model=models[120],
		img1=img1,
		img2=img2,
		num_steps=5,
		device=device
	)

	fig, axes = plt.subplots(1, len(interpolated), figsize=(15, 3))

	for i, img in enumerate(interpolated):
		alpha = i / (len(interpolated) - 1)
		axes[i].imshow(img[0].permute(1, 2, 0))
		axes[i].set_title(f"α={alpha:.2f}")
		axes[i].axis('off')

	plt.suptitle("Интерполяция в латентном пространстве")
	plt.tight_layout()
	plt.show()


