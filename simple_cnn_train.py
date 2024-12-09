# 필요한 라이브러리 임포트
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import seaborn as sns
import matplotlib.pyplot as plt

# 기본 설정
BATCH_SIZE = 32  # 배치 크기
NUM_EPOCHS = 10  # 학습 에폭 수
LEARNING_RATE = 2e-4  # 학습률
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 디바이스 설정
PREPROCESS_DIR = 'data/preprocess_data'  # 전처리된 데이터 경로
PATIENCE = 5  # Early stopping patience

# Custom CNN 모델 정의
class SimplifiedCNN(nn.Module):
    """간단한 CNN 모델로 이미지를 분류."""
    def __init__(self, num_classes):
        super(SimplifiedCNN, self).__init__()
        # Convolutional Layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  # Max Pooling
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        # Fully Connected Layers
        self.fc1 = nn.Linear(32 * 56 * 56, 64)  # 첫 번째 FC Layer
        self.fc2 = nn.Linear(64, num_classes)  # 두 번째 FC Layer

    def forward(self, x):
        # Convolution + ReLU + Pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))  # Fully Connected Layer 1
        x = self.fc2(x)  # Fully Connected Layer 2 (출력)
        return x

# 데이터셋 클래스 정의
class CustomImageDataset(Dataset):
    """이미지 데이터셋을 정의. 각 이미지와 레이블 반환."""
    def __init__(self, root_dir, split='train'):
        self.root_dir = os.path.join(root_dir, split)  # 데이터 분할 경로
        self.image_paths = []  # 이미지 경로 리스트
        self.labels = []  # 레이블 리스트
        self.class_to_idx = {}  # 클래스 이름과 인덱스 매핑

        # 클래스별로 이미지 경로 및 레이블 수집
        for idx, class_name in enumerate(sorted(os.listdir(self.root_dir))):
            self.class_to_idx[class_name] = idx
            class_dir = os.path.join(self.root_dir, class_name)
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.endswith('.jpg'):
                        self.image_paths.append(os.path.join(class_dir, img_name))
                        self.labels.append(idx)

        # 데이터 변환 파이프라인 정의
        self.transform = Compose([
            Resize((224, 224)),  # 이미지 크기 조정
            ToTensor(),  # 텐서로 변환
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 정규화
        ])

    def __len__(self):
        return len(self.image_paths)  # 데이터셋 크기 반환

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')  # 이미지 로드 및 RGB로 변환
        image = self.transform(image)  # 변환 적용
        label = self.labels[idx]  # 레이블
        return image, label

# 학습 루프
def train_one_epoch(model, train_loader, optimizer, criterion, device):
    """한 에폭 동안 모델 학습."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    progress_bar = tqdm(train_loader, desc='Training')
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)

        # 옵티마이저 초기화 및 순전파
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()  # 역전파
        optimizer.step()  # 매개변수 업데이트

        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        progress_bar.set_postfix({'loss': loss.item()})

    epoch_loss = total_loss / len(train_loader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc

# 검증 루프
def evaluate(model, val_loader, criterion, device):
    """모델 평가."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Evaluating'):
            images, labels = images.to(device), labels.to(device)
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

# Early Stopping 클래스 정의
class EarlyStopping:
    """검증 손실 개선 여부에 따라 학습 중단."""
    def __init__(self, patience=7, min_delta=0, path='checkpoint.pth'):
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, model):
        """최고 성능 모델 저장."""
        torch.save(model.state_dict(), self.path)

# 메인 함수
def main():
    print(f"Using device: {DEVICE}")

    # 데이터셋 및 데이터 로더 초기화
    train_dataset = CustomImageDataset(PREPROCESS_DIR, split='train')
    val_dataset = CustomImageDataset(PREPROCESS_DIR, split='val')
    test_dataset = CustomImageDataset(PREPROCESS_DIR, split='test')

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 모델 초기화
    num_classes = len(os.listdir(os.path.join(PREPROCESS_DIR, 'train')))
    model = SimplifiedCNN(num_classes).to(DEVICE)

    # 손실 함수 및 옵티마이저 정의
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    early_stopping = EarlyStopping(patience=PATIENCE, path='best_model.pth')

    # 학습 루프
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, criterion, DEVICE)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    # 테스트
    model.load_state_dict(torch.load('best_model.pth'))
    test_loss, test_acc, test_f1, _, _ = evaluate(model, test_loader, criterion, DEVICE)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Test F1: {test_f1:.4f}")


if __name__ == "__main__":
    main()
