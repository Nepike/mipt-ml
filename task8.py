import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def generate_points(n, R=1.0):
	theta = torch.acos(2 * torch.rand(n, 1) - 1)  # [0, π]
	phi = 2 * torch.pi * torch.rand(n, 1)  # [0, 2π]

	x = R * torch.sin(theta) * torch.cos(phi)
	y = R * torch.sin(theta) * torch.sin(phi)
	z = R * torch.cos(theta)

	points = torch.cat([x, y, z], dim=1)
	return points.to(device)


class Generator(nn.Module):
	def __init__(self, latent_dim, hidden_dim, output_dim=2):
		super(Generator, self).__init__()
		self.net = nn.Sequential(
			nn.Linear(latent_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, output_dim),
			nn.Tanh()
		)

	def forward(self, z):
		angles = self.net(z)  # angles: [batch_size, 2]
		theta = (angles[:, 0] + 1) * torch.pi / 2  # [0, π]
		phi = (angles[:, 1] + 1) * torch.pi  # [0, 2π]

		x = R * torch.sin(theta) * torch.cos(phi)
		y = R * torch.sin(theta) * torch.sin(phi)
		z_coord = R * torch.cos(theta)

		points = torch.stack([x, y, z_coord], dim=1)
		return points


class Discriminator(nn.Module):
	def __init__(self, input_dim=3, hidden_dim=128):
		super(Discriminator, self).__init__()
		self.net = nn.Sequential(
			nn.Linear(input_dim, hidden_dim),
			nn.LeakyReLU(0.2),
			nn.Linear(hidden_dim, hidden_dim),
			nn.LeakyReLU(0.2),
			nn.Linear(hidden_dim, 1),
			nn.Sigmoid()
		)

	def forward(self, x):
		return self.net(x)


if __name__ == '__main__':
	R = 1.0
	latent_dim = 10
	hidden_dim = 128
	batch_size = 64
	epochs = 600
	lr = 0.0002
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	generator = Generator(latent_dim, hidden_dim).to(device)
	discriminator = Discriminator().to(device)

	optimizer_G = optim.Adam(generator.parameters(), lr=lr)
	optimizer_D = optim.Adam(discriminator.parameters(), lr=lr)

	criterion = nn.BCELoss()

	losses_G = []
	losses_D = []

	for epoch in range(epochs):
		for _ in range(100):
			optimizer_D.zero_grad()

			real = generate_points(batch_size, R)
			real_labels = torch.ones(batch_size, 1).to(device)
			real_output = discriminator(real)
			d_loss_real = criterion(real_output, real_labels)

			z = torch.randn(batch_size, latent_dim).to(device)
			fake = generator(z)
			fake_labels = torch.zeros(batch_size, 1).to(device)
			fake_output = discriminator(fake.detach())
			d_loss_fake = criterion(fake_output, fake_labels)

			d_loss = d_loss_real + d_loss_fake
			d_loss.backward()

			optimizer_D.step()

			optimizer_G.zero_grad()
			z = torch.randn(batch_size, latent_dim).to(device)
			fake = generator(z)
			output = discriminator(fake)
			g_loss = criterion(output, real_labels)  # пытаемся обмануть D

			g_loss.backward()
			optimizer_G.step()

		losses_G.append(g_loss.item())
		losses_D.append(d_loss.item())

		if epoch % 50 == 0:
			print(f"Epoch {epoch}, D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")


	# ВИЗУАЛИЗАЦИЯ
	z = torch.randn(1000, latent_dim).to(device)
	generated_points = generator(z).detach().cpu().numpy()
	real_points_viz = generate_points(500, R).cpu().numpy()

	fig = plt.figure(figsize=(16, 9))

	ax1 = fig.add_subplot(121, projection='3d')
	ax1.scatter(generated_points[:, 0], generated_points[:, 1], generated_points[:, 2], alpha=0.6)
	ax1.set_title("Generated Points on Sphere")
	ax1.set_xlabel('X')
	ax1.set_ylabel('Y')
	ax1.set_zlabel('Z')

	ax2 = fig.add_subplot(122, projection='3d')
	ax2.scatter(real_points_viz[:, 0], real_points_viz[:, 1], real_points_viz[:, 2], alpha=0.6)
	ax2.set_title("Real Points on Sphere")
	ax2.set_xlabel('X')
	ax2.set_ylabel('Y')
	ax2.set_zlabel('Z')

	plt.tight_layout()
	plt.show()

	plt.figure(figsize=(10, 5))
	plt.plot(losses_G, label='Generator Loss')
	plt.plot(losses_D, label='Discriminator Loss')
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.legend()
	plt.title('GAN Training Losses')
	plt.show()