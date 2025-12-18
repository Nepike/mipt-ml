import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


class MLP(nn.Module):
	def __init__(self, input_layer_size):
		super(MLP, self).__init__()

		self.fc1 = nn.Linear(input_layer_size, 128)
		self.fc2 = nn.Linear(128, 64)
		self.fc3 = nn.Linear(64, 2)

		self.dropout = nn.Dropout(0.3)
		self.relu = nn.ReLU()

	def forward(self, x):
		x = self.fc1(x)
		x = self.relu(x)
		x = self.dropout(x)

		x = self.fc2(x)
		x = self.relu(x)
		x = self.dropout(x)

		x = self.fc3(x)
		return x


class CNN(nn.Module):
	def __init__(self, num_kernels):
		super(CNN, self).__init__()

		self.conv1 = nn.Conv2d(3, num_kernels, 3, padding=1)
		self.bn1 = nn.BatchNorm2d(num_kernels)

		self.conv2 = nn.Conv2d(num_kernels, num_kernels, 3, padding=1)
		self.bn2 = nn.BatchNorm2d(num_kernels)

		self.conv3 = nn.Conv2d(num_kernels, num_kernels, 3, padding=1)
		self.bn3 = nn.BatchNorm2d(num_kernels)

		self.conv4 = nn.Conv2d(num_kernels, num_kernels, 3, padding=1)
		self.bn4 = nn.BatchNorm2d(num_kernels)

		self.pool = nn.MaxPool2d(5, stride=2, padding=2)
		self.relu = nn.ReLU()

		# по идее на выходе размер:
		# 32x32 -> 16x16 -> 8x8 -> 4x4 -> 2x2
		# *num_kernels ядер = 2x2xnum_kernels
		self.mlp = MLP(input_layer_size=num_kernels*4)

	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.pool(x)

		x = self.conv2(x)
		x = self.bn2(x)
		x = self.relu(x)
		x = self.pool(x)

		x = self.conv3(x)
		x = self.bn3(x)
		x = self.relu(x)
		x = self.pool(x)

		x = self.conv4(x)
		x = self.bn4(x)
		x = self.relu(x)
		x = self.pool(x)

		x = x.view(x.size(0), -1)
		x = self.mlp(x)
		return x


class RCNN(CNN):
	def __init__(self, num_kernels):
		super(RCNN, self).__init__(num_kernels)

		self.skip1 = nn.Sequential(
			nn.Conv2d(3, num_kernels, 1, bias=False),
			nn.BatchNorm2d(num_kernels)
		)

		# теперь размер:
		# 32x32 -> 16x16 -> 8x8
		# *num_kernels ядер = 8x8xnum_kernels
		self.mlp = MLP(input_layer_size=num_kernels * 64)

	def forward(self, x):
		remember = self.skip1(x)
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)

		x = self.conv2(x)
		x = self.bn2(x)

		x += remember
		x = self.relu(x)
		x = self.pool(x)

		remember = x

		x = self.conv3(x)
		x = self.bn3(x)
		x = self.relu(x)

		x = self.conv4(x)
		x = self.bn4(x)

		x += remember
		x = self.relu(x)
		x = self.pool(x)

		x = x.view(x.size(0), -1)
		x = self.mlp(x)
		return x


def filter_classes(dataset, class1=0, class2=1):
	targets = np.array(dataset.targets)
	indices = np.where((targets == class1) | (targets == class2))[0]
	return Subset(dataset, indices)


if __name__ == "__main__":
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	batch_size = 256
	num_epochs = 100

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
		batch_size=batch_size,
		shuffle=True,
		num_workers=4,
		pin_memory=True,
		persistent_workers=True
	)

	test_loader = DataLoader(
		test_dataset,
		batch_size=batch_size,
		shuffle=False,
		num_workers=4,
		pin_memory=True,
		persistent_workers=True
	)

	kernels_nums = [8, 16, 32, 64]

	for kernels_num in kernels_nums:
		model = CNN(num_kernels=kernels_num).to(device)

		criterion = nn.CrossEntropyLoss()
		optimizer = optim.Adam(model.parameters(), lr=0.001)

		print(f"Начало обучения модели с {kernels_num} ядрами")
		start_time = time.time()
		for epoch in range(num_epochs):
			model.train()
			train_loss = 0.0

			for batch_idx, (inputs, labels) in enumerate(train_loader):
				inputs, labels = inputs.to(device), labels.to(device)

				optimizer.zero_grad()
				outputs = model(inputs)
				loss = criterion(outputs, labels)
				loss.backward()
				optimizer.step()

				train_loss += loss.item()

			avg_train_loss = train_loss / len(train_loader)

			model.eval()
			test_loss = 0.0
			correct = 0
			total = 0
			with torch.no_grad():
				for inputs, labels in test_loader:
					inputs, labels = inputs.to(device), labels.to(device)

					outputs = model(inputs)
					loss = criterion(outputs, labels)
					test_loss += loss.item()

					_, predicted = torch.max(outputs.data, 1)
					total += labels.size(0)
					correct += (predicted == labels).sum().item()

			avg_test_loss = test_loss / len(test_loader)
			accuracy = 100 * correct / total

			if (epoch % 10 == 0) or (epoch == num_epochs - 1):
				print(f'Epoch [{epoch + 1}/{num_epochs}], '
					  f'Train Loss: {avg_train_loss:.4f}, '
					  f'Test Loss: {avg_test_loss:.4f}, '
					  f'Accuracy: {accuracy:.2f}%')

		total_time = time.time() - start_time
		print(f"Обучение модели с {kernels_num} ядрами звершено за {total_time: .2f} сек.")
		print("-"*64)

		# PATH = f'./models/cnn_{kernels_num}.pth'
		# torch.save(model.state_dict(), PATH)

	# ----------------------------------------------------------------------------------
	# ТЕПРЬ ОБУЧАЕМ RCNN
	# ----------------------------------------------------------------------------------

	kernels_nums = [16]

	for kernels_num in kernels_nums:
		model = RCNN(num_kernels=kernels_num).to(device)

		criterion = nn.CrossEntropyLoss()
		optimizer = optim.Adam(model.parameters(), lr=0.001)

		print(f"Начало обучения ОСТАТОЧНОЙ модели с {kernels_num} ядрами")
		start_time = time.time()
		for epoch in range(num_epochs):
			model.train()
			train_loss = 0.0

			for batch_idx, (inputs, labels) in enumerate(train_loader):
				inputs, labels = inputs.to(device), labels.to(device)

				optimizer.zero_grad()
				outputs = model(inputs)
				loss = criterion(outputs, labels)
				loss.backward()
				optimizer.step()

				train_loss += loss.item()

			avg_train_loss = train_loss / len(train_loader)

			model.eval()
			test_loss = 0.0
			correct = 0
			total = 0
			with torch.no_grad():
				for inputs, labels in test_loader:
					inputs, labels = inputs.to(device), labels.to(device)

					outputs = model(inputs)
					loss = criterion(outputs, labels)
					test_loss += loss.item()

					_, predicted = torch.max(outputs.data, 1)
					total += labels.size(0)
					correct += (predicted == labels).sum().item()

			avg_test_loss = test_loss / len(test_loader)
			accuracy = 100 * correct / total

			if (epoch % 10 == 0) or (epoch == num_epochs - 1):
				print(f'Epoch [{epoch + 1}/{num_epochs}], '
					  f'Train Loss: {avg_train_loss:.4f}, '
					  f'Test Loss: {avg_test_loss:.4f}, '
					  f'Accuracy: {accuracy:.2f}%')

		total_time = time.time() - start_time
		print(f"Обучение остатоночй модели с {kernels_num} ядрами звершено за {total_time: .2f} сек.")
		print("-" * 64)




