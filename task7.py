import os
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
from tqdm import tqdm


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        """
        d_model: размерность эмбеддингов (256)
        max_len: максимальная длина последовательности
        """
        super().__init__()

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # x имеет форму [batch_size, seq_len, d_model]
        # Добавляем позиционное кодирование для первых seq_len позиций
        x = x + self.pe[:, :x.size(1)]
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        """
        d_model: размерность эмбеддингов (256)
        num_heads: количество головок внимания (8)
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Размерность каждой головки

        # Линейные слои для Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        Q = self.W_q(query)  # [batch_size, seq_len, d_model]
        K = self.W_k(key)  # [batch_size, seq_len, d_model]
        V = self.W_v(value)  # [batch_size, seq_len, d_model]

        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Вычисляем attention scores: Q * K^T / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)

        output = torch.matmul(attn_weights, V)

        output = output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )

        output = self.W_o(output)

        return output, attn_weights


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=1024):
        """
        d_model: размерность эмбеддингов (256)
        d_ff: размерность скрытого слоя
        """
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.linear1(x)  # [batch_size, seq_len, d_ff]
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, mask):

        attn_output, _ = self.self_attn(x, x, x, mask)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)

        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)

        return x


class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model=256, num_layers=6, num_heads=8, max_len=128):
        """
        vocab_size: размер словаря
        d_model: размерность эмбеддингов (256)
        num_layers: количество блоков декодера (6)
        num_heads: количество головок внимания (8)
        max_len: максимальная длина последовательности (128)
        """
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, d_model)

        self.positional_encoding = PositionalEncoding(d_model, max_len)

        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(d_model, num_heads) for _ in range(num_layers)
        ])

        self.output_layer = nn.Linear(d_model, vocab_size)

        self.dropout = nn.Dropout(0.3)

    def create_mask(self, seq_len, device):
        mask = torch.tril(
            torch.ones(seq_len, seq_len, device=device)
        )
        mask = mask.unsqueeze(0).unsqueeze(0)
        return mask

    def forward(self, x):
        batch_size, seq_len = x.shape

        x = self.token_embedding(x)  # [batch_size, seq_len, d_model]
        x = self.dropout(x)
        x = self.positional_encoding(x)

        mask = self.create_mask(seq_len, x.device)

        for block in self.decoder_blocks:
            x = block(x, mask)

        logits = self.output_layer(x)  # [batch_size, seq_len, vocab_size]

        return logits

    def generate(self, prompt, max_length=50, temperature=1.0, device='cpu'):
        self.eval()

        if isinstance(prompt, str):
            prompt_indices = [self.char_to_idx.get(ch, 0) for ch in prompt]
        else:
            prompt_indices = prompt

        generated = prompt_indices.copy()

        with torch.no_grad():
            for _ in range(max_length):
                input_seq = generated[-128:] if len(generated) > 128 else generated
                input_tensor = torch.tensor([input_seq]).to(device)

                logits = self(input_tensor)  # [1, seq_len, vocab_size]

                last_logits = logits[0, -1, :] / temperature  # [vocab_size]

                probs = F.softmax(last_logits, dim=-1)

                next_token = torch.multinomial(probs, 1).item()

                generated.append(next_token)

                if next_token == self.eos_token:
                    break

        return generated


class TextDataset(Dataset):
    def __init__(self, text, seq_length=128):
        """
        text: исходный текст
        seq_length: длина последовательности для обучения
        """
        self.text = text
        self.seq_length = seq_length

        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)

        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}

        self.data = [self.char_to_idx[ch] for ch in text]

        self.sequences = []
        for i in range(0, len(self.data) - seq_length):
            input_seq = self.data[i:i + seq_length]
            target_seq = self.data[i + 1:i + seq_length + 1]
            self.sequences.append((input_seq, target_seq))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        input_seq, target_seq = self.sequences[idx]
        return torch.tensor(input_seq), torch.tensor(target_seq)


def load_text_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()

    return text


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_tokens = 0

    progress_bar = tqdm(dataloader, desc="Обучение", leave=True)

    for batch_idx, (inputs, targets) in enumerate(progress_bar):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)  # [batch_size, seq_len, vocab_size]

        logits = logits.view(-1, logits.size(-1))  # [batch_size*seq_len, vocab_size]
        targets = targets.view(-1)  # [batch_size*seq_len]

        loss = criterion(logits, targets)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_tokens += targets.size(0)

        avg_loss = total_loss / (batch_idx + 1)
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{avg_loss:.4f}',
            'tokens/s': f'{total_tokens / (batch_idx + 1):.0f}'
        })

    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            logits = model(inputs)
            logits = logits.view(-1, logits.size(-1))
            targets = targets.view(-1)

            loss = criterion(logits, targets)
            total_loss += loss.item()

    return total_loss / len(dataloader)


def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, vocab_size, checkpoint_dir='checkpoints'):
    """
    Сохраняет полный чекпоинт модели

    Args:
        model: модель трансформера
        optimizer: оптимизатор
        epoch: текущая эпоха
        train_loss: потери на обучении
        val_loss: потери на валидации
        vocab_size: размер словаря
        checkpoint_dir: директория для сохранения
    """
    # Создаем директорию, если не существует
    Path(checkpoint_dir).mkdir(exist_ok=True)

    # Формируем имя файла с timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_path = Path(checkpoint_dir) / f'transformer_epoch_{epoch}_{timestamp}.pt'

    # Сохраняем всё
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'vocab_size': vocab_size,
        'd_model': model.d_model if hasattr(model, 'd_model') else 256,
        'num_layers': len(model.decoder_blocks),
        'num_heads': model.decoder_blocks[0].self_attn.num_heads if hasattr(model.decoder_blocks[0].self_attn,
                                                                            'num_heads') else 8,
        'char_to_idx': model.char_to_idx,
        'idx_to_char': model.idx_to_char,
        'eos_token': model.eos_token,
    }, checkpoint_path)

    print(f"Чекпоинт сохранен: {checkpoint_path}")

    # Также сохраняем последний чекпоинт отдельно
    last_checkpoint_path = Path(checkpoint_dir) / '7_last_checkpoint.pt'
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'char_to_idx': model.char_to_idx,
        'idx_to_char': model.idx_to_char,
    }, last_checkpoint_path)

    return checkpoint_path


def load_checkpoint(checkpoint_path, model, optimizer=None, device='cpu'):
    """
    Загружает чекпоинт модели

    Args:
        checkpoint_path: путь к чекпоинту
        model: модель для загрузки весов
        optimizer: опционально, оптимизатор для загрузки состояния
        device: устройство для загрузки

    Returns:
        epoch: номер эпохи
        train_loss: потери на обучении
        val_loss: потери на валидации
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Чекпоинт не найден: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Загружаем состояние модели
    model.load_state_dict(checkpoint['model_state_dict'])

    # Загружаем состояние оптимизатора (если передан)
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Загружаем словари
    if 'char_to_idx' in checkpoint:
        model.char_to_idx = checkpoint['char_to_idx']
        model.idx_to_char = checkpoint['idx_to_char']
        model.eos_token = checkpoint.get('eos_token', 0)

    print(f"Чекпоинт загружен: {checkpoint_path}")
    print(f"Эпоха: {checkpoint.get('epoch', 0)}")
    print(f"Train Loss: {checkpoint.get('train_loss', 0):.4f}")
    print(f"Val Loss: {checkpoint.get('val_loss', 0):.4f}")

    return (
        checkpoint.get('epoch', 0),
        checkpoint.get('train_loss', 0),
        checkpoint.get('val_loss', 0)
    )


if __name__ == '__main__':
    if torch.cuda.is_available():
        # Проверяем, есть ли CUDA и достаточная память
        device = torch.device('cuda')
        # Используем Tensor Cores на RTX 3060 если доступно
        torch.backends.cuda.matmul.allow_tf32 = True  # Для ускорения матричных операций
        torch.backends.cudnn.benchmark = True  # Автонастройка cuDNN
    else:
        device = torch.device('cpu')



    BATCH_SIZE = 256
    NUM_WORKERS = 3
    PIN_MEMORY = True if device.type == 'cuda' else False
    CHECKPOINT_DIR = './models'

    train_text = load_text_data('./wikitext-2/train.txt')
    train_text = train_text[:len(train_text)//2]

    train_dataset = TextDataset(train_text, seq_length=128)
    vocab_size = train_dataset.vocab_size

    val_text = load_text_data('./wikitext-2/test.txt')
    #val_text = val_text[:len(val_text)//320]
    val_dataset = TextDataset(val_text, seq_length=128)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        prefetch_factor=2
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        prefetch_factor=2
    )


    model = TransformerLM(
        vocab_size=vocab_size,
        d_model=256,
        num_layers=6,
        num_heads=8,
        max_len=128
    )

    model = model.to(device)

    model.char_to_idx = train_dataset.char_to_idx
    model.idx_to_char = train_dataset.idx_to_char
    model.eos_token = train_dataset.char_to_idx.get('.', 0)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01
    )

    num_epochs = 0
    start_epoch = 0
    best_val_loss = float('inf')

    last_checkpoint_path = Path(CHECKPOINT_DIR) / '7_last_checkpoint.pt'
    if last_checkpoint_path.exists():
        print(f"Найден последний чекпоинт: {last_checkpoint_path}")
        response = input("Загрузить чекпоинт? (y/n): ")
        if response.lower() == 'y':
            start_epoch, _, best_val_loss = load_checkpoint(
                last_checkpoint_path, model, optimizer, device
            )
            start_epoch += 1
            print(f"Продолжаем обучение с эпохи {start_epoch}")


    for epoch in range(start_epoch, num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

        val_loss = evaluate(model, val_loader, criterion, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, epoch, train_loss, val_loss,
                vocab_size, CHECKPOINT_DIR
            )
            print(f"✨ Новая лучшая модель! Val Loss: {val_loss:.4f}")

            # Сохранение чекпоинта каждые 5 эпох
        elif (epoch + 1) % 5 == 0:
            save_checkpoint(
                model, optimizer, epoch, train_loss, val_loss,
                vocab_size, CHECKPOINT_DIR
            )

        print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

        if epoch > 10 and val_loss > best_val_loss * 1.1:  # Ухудшение на 10%
            print(f"Ранняя остановка: нет прогресса")
            break

    prompts = ["This phenomenon was", "Space-time", "Artificial", "Japan", "World War II was"]
    for prompt in prompts:
        print(f"\nПромпт: '{prompt}'")

        generated_indices = model.generate(
            prompt=prompt,
            max_length=5000,
            temperature=0.9,
            device=device
        )

        generated_text = ''.join([train_dataset.idx_to_char[idx] for idx in generated_indices])

        print(f"Результат: {generated_text[:7000]}...")



