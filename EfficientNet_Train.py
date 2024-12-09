import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import timm
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import random

# 기본 설정
BATCH_SIZE = 16
NUM_EPOCHS = 30
LEARNING_RATE = 1e-4  # EfficientNet 학습률
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_NAME = "efficientnet_b0"
PREPROCESS_DIR = 'data/preprocess_data'
PATIENCE = 5
SEED = 42

def seed_everything(seed):
    """재현성을 위한 시드 고정 함수"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, split='train'):
        self.root_dir = os.path.join(root_dir, split)
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}

        # 클래스 및 이미지 경로 수집
        for idx, class_name in enumerate(sorted(os.listdir(self.root_dir))):
            self.class_to_idx[class_name] = idx
            class_dir = os.path.join(self.root_dir, class_name)
            if os.path.isdir(class_dir):
                for img_name in sorted(os.listdir(class_dir)):
                    if img_name.endswith('.jpg'):
                        self.image_paths.append(os.path.join(class_dir, img_name))
                        self.labels.append(idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        image = image.resize((224, 224), Image.BILINEAR)
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        # ImageNet 정규화 적용
        mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        image = (image - mean) / std
        label = self.labels[idx]
        return image, label

def train_one_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    progress_bar = tqdm(train_loader, desc='Training')
    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

        progress_bar.set_postfix({'loss': loss.item()})

    epoch_loss = total_loss / len(train_loader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc

def plot_confusion_matrix(cm, class_names, title='Confusion Matrix'):
    """혼동 행렬 시각화"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'{title.lower().replace(" ", "_")}.png')
    plt.close()

class EarlyStopping:
    """Early stopping 구현"""
    def __init__(self, patience=7, min_delta=0, path='checkpoint.pth'):
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model, optimizer, epoch, val_acc):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model, optimizer, epoch, val_acc)
        elif val_loss > self.best_loss + self.min_delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model, optimizer, epoch, val_acc)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, optimizer, epoch, val_acc):
        """모델 체크포인트 저장"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc
        }, self.path)

def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Evaluating'):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    val_loss = total_loss / len(val_loader)
    val_acc = accuracy_score(all_labels, all_preds)
    val_f1 = f1_score(all_labels, all_preds, average='weighted')

    return val_loss, val_acc, val_f1, all_preds, all_labels

def plot_misclassified_images(images, true_labels, pred_labels, class_names, num_images=10):
    """오분류된 이미지 시각화"""
    n = min(len(images), num_images)
    if n == 0:
        print("No misclassified images found.")
        return

    fig = plt.figure(figsize=(20, 4))
    for idx in range(n):
        ax = plt.subplot(1, n, idx + 1)
        img = images[idx].permute(1, 2, 0).cpu().numpy()
        img = np.clip(img, 0, 1)
        plt.imshow(img)
        plt.title(f'{class_names[true_labels[idx]]}\n→\n{class_names[pred_labels[idx]]}')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('misclassified_examples.png')
    plt.show

def test(model, test_loader, criterion, device, idx_to_class):
    """테스트 데이터에 대한 최종 평가"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    misclassified_images = []
    misclassified_true = []
    misclassified_pred = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Testing'):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)

            # 오분류된 이미지 수집
            misclassified_mask = preds != labels
            if misclassified_mask.any():
                misclassified_images.extend(images[misclassified_mask])
                misclassified_true.extend(labels[misclassified_mask].cpu().numpy())
                misclassified_pred.extend(preds[misclassified_mask].cpu().numpy())

            preds = preds.cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    test_loss = total_loss / len(test_loader)
    test_acc = accuracy_score(all_labels, all_preds)
    test_f1 = f1_score(all_labels, all_preds, average='weighted')

    # 분류 리포트 생성
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    report = classification_report(all_labels, all_preds, target_names=class_names)

    # 혼동 행렬 생성 및 시각화
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, class_names, title='Test Confusion Matrix')

    # 오분류된 이미지 시각화
    plot_misclassified_images(
        misclassified_images,
        misclassified_true,
        misclassified_pred,
        class_names
    )

    return test_loss, test_acc, test_f1, report

def main():
    # 시드 고정
    seed_everything(SEED)
    print(f"Using device: {DEVICE}")

    # EfficientNet 모델 로드
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=len(os.listdir(os.path.join(PREPROCESS_DIR, 'train'))))
    model = model.to(DEVICE)

    # 데이터셋 및 데이터로더 생성
    train_dataset = CustomImageDataset(PREPROCESS_DIR, split='train')
    val_dataset = CustomImageDataset(PREPROCESS_DIR, split='val')
    test_dataset = CustomImageDataset(PREPROCESS_DIR, split='test')

    # Worker 초기화 함수 정의
    def worker_init_fn(worker_id):
        worker_seed = SEED + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    # DataLoader 생성
    generator = torch.Generator()
    generator.manual_seed(SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        worker_init_fn=worker_init_fn,
        generator=generator
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        worker_init_fn=worker_init_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        worker_init_fn=worker_init_fn
    )

    # Loss function과 optimizer 설정
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=3,
        verbose=True
    )

    # Early stopping 초기화
    early_stopping = EarlyStopping(patience=PATIENCE, path='best_model.pth')

    # 클래스 이름 매핑
    idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}

    # 학습 루프
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")

        # 학습
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)

        # 검증
        val_loss, val_acc, val_f1, val_preds, val_labels = evaluate(model, val_loader, criterion, DEVICE)

        # Learning rate 업데이트
        scheduler.step(val_loss)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
        print(f"Current LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Early stopping 체크
        early_stopping(val_loss, model, optimizer, epoch, val_acc)

        # 현재 에폭의 혼동 행렬 생성 및 저장
        class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
        cm = confusion_matrix(val_labels, val_preds)
        plot_confusion_matrix(cm, class_names, title=f'Validation Confusion Matrix Epoch {epoch+1}')

        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    # 최고 성능 모델 로드
    checkpoint = torch.load('best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    # 테스트 데이터로 최종 평가
    test_loss, test_acc, test_f1, test_report = test(model, test_loader, criterion, DEVICE, idx_to_class)
    print("\nTest Results:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")
    print("\nTest Classification Report:")
    print(test_report)

if __name__ == "__main__":
    main()