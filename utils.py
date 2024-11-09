import re
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import numpy as np
from scipy.linalg import lstsq

def get_embedding(word, glove_model, oov_vectors):
    """
    Retrieve the GloVe embedding for a word. If the word is OOV, return a random embedding,
    storing it in oov_vectors if it does not already exist.
    """
    
    if word in glove_model:
        return glove_model[word]
    else:
        # Check if OOV embedding already exists, else create and store it
        if word not in oov_vectors:
            oov_vectors[word] = np.random.normal(size=300)
        return oov_vectors[word]

def get_improved_embedding(word, glove_model, fasttext_model, oov_vectors):
    """
    Retrieve the embedding for a word from GloVe, or from FastText (transformed to GloVe space),
    or generate a random embedding if OOV in both, storing it in oov_vectors if new.
    """

    if word in glove_model:
        return glove_model[word]
    
    # Compute the FastText-to-GloVe transformation matrix only once per session
    if not hasattr(get_improved_embedding, "W_fasttext"):
        # Find common words in both FastText and GloVe models
        common_words = list(set(glove_model.key_to_index).intersection(set(fasttext_model.key_to_index)))
        X_fasttext = np.array([fasttext_model[w] for w in common_words])
        Y_glove = np.array([glove_model[w] for w in common_words])
        get_improved_embedding.W_fasttext, _, _, _ = lstsq(X_fasttext, Y_glove)
        

    # If the word is in FastText, transform its embedding to GloVe space
    if word in fasttext_model:
        transformed_embedding = np.dot(fasttext_model[word], get_improved_embedding.W_fasttext)
        return transformed_embedding
    else:
        # Check if OOV embedding already exists, else create and store it
        if word not in oov_vectors:
            oov_vectors[word] = np.random.normal(size=300)
        return oov_vectors[word]


class SentimentDataset(Dataset):
    def __init__(self, texts, labels, glove_model, oov_vectors, max_length=50):
        self.texts = texts
        self.labels = labels
        self.glove_model = glove_model
        self.oov_vectors = oov_vectors
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def tokenize(self, text):
        return re.findall(r'\b\w+\b', text.lower())

    def get_vector(self, word):
        if word in self.glove_model:
            return self.glove_model[word]
        elif word in self.oov_vectors:
            return self.oov_vectors[word]
        else:
            return np.zeros(self.glove_model.vector_size)

    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = self.tokenize(text)
        tokens = tokens[:self.max_length] + [''] * (self.max_length - len(tokens))
        vectors = [self.get_vector(token) for token in tokens]
        return torch.FloatTensor(vectors), torch.LongTensor([self.labels[idx]])
    

class EnhancedEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, pretrained_embeddings):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(
            torch.FloatTensor(pretrained_embeddings),
            padding_idx=0,
            freeze=False
        )

    def forward(self, x):
        return self.embedding(x)

def get_device():
    device_name = ''
    if torch.cuda.is_available():
        device_name = 'cuda'
    elif torch.mps.is_available():
        device_name = 'mps'
    else:
        device_name = 'cpu'
    print("DEVICE: ", device_name)
    return torch.device(device_name)

def train_model(model, train_loader, val_loader, num_epochs, device, model_name):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=5,
        T_mult=2,
        eta_min=1e-6
    )

    best_val_acc = 0
    patience = 7
    patience_counter = 0

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_acc': [],
        'lr': []
    }

    ema = torch.optim.swa_utils.AveragedModel(model)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.squeeze().to(device)

            if epoch > 3:
                alpha = 0.2
                lam = np.random.beta(alpha, alpha)
                index = torch.randperm(inputs.size(0)).to(device)
                mixed_inputs = lam * inputs + (1 - lam) * inputs[index]
                inputs = mixed_inputs

            optimizer.zero_grad()
            outputs = model(inputs)

            if epoch > 3:
                loss = lam * criterion(outputs, labels) + (1 - lam) * criterion(outputs, labels[index])
            else:
                loss = criterion(outputs, labels)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            ema.update_parameters(model)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = 100. * correct / total

        ema.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.squeeze().to(device)

                outputs = ema(inputs)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100. * val_correct / val_total

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        history['train_loss'].append(total_loss/len(train_loader))
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)

        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Training Loss: {total_loss/len(train_loader):.4f}')
        print(f'Training Accuracy: {train_acc:.2f}%')
        print(f'Validation Accuracy: {val_acc:.2f}%')
        print(f'Learning Rate: {current_lr:.6f}')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            print(f'New best validation accuracy! Saving model...')
            torch.save({
                'epoch': epoch,
                'model_state_dict': ema.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc,
                'history': history
            }, f'best_{model_name}.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping triggered after epoch {epoch+1}')
                break

    return history