# 필요한 라이브러리 임포트
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pickle
import os
import cv2
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import transforms
from tqdm import tqdm
import warnings

from feature_extract import FEATURE_DIR

# 경고 메시지 무시
warnings.filterwarnings('ignore')

# Config 클래스 정의: 하이퍼파라미터 및 모델 저장 디렉토리 설정
class Config:
    IMAGE_SIZE = 256  # 입력 이미지 크기
    BATCH_SIZE = 8  # 학습 시 한 번에 처리할 배치 크기
    NUM_WORKERS = 2  # 데이터 로딩 시 병렬 프로세스 개수

    DROPOUT_RATE = 0.4  # 드롭아웃 확률
    HIDDEN_DIM = 256  # 히든 레이어 차원 수
    BN_MOMENTUM = 0.1  # Batch Normalization 모멘텀

    LEARNING_RATE = 0.0001  # 초기 학습률
    NUM_EPOCHS = 150  # 학습 에폭 수
    EARLY_STOP_PATIENCE = 10  # 얼리 스톱 기준: 성능 개선이 없을 때 기다리는 에폭 수
    WEIGHT_DECAY = 0.001  # 가중치 감소 (L2 정규화)

    LABEL_SMOOTHING = 0.1  # 레이블 스무딩 비율
    GRAD_CLIP = 1.0  # 그래디언트 클리핑 값

    WARMUP_EPOCHS = 5  # 워밍업 에폭 수
    USE_SCHEDULER = True  # 학습률 스케줄러 사용 여부
    SCHEDULER_PATIENCE = 5  # 스케줄러 참을성
    SCHEDULER_FACTOR = 0.3  # 스케줄러 감소 비율
    MIN_LR = 1e-6  # 스케줄러 최소 학습률

    SPARSITY_WEIGHT = 0.0005  # 희소성 규제 가중치
    SAVE_DIR = 'model_checkpoints'  # 모델 저장 디렉토리
    os.makedirs(SAVE_DIR, exist_ok=True)

# 데이터셋 클래스 정의: 이미지 및 특징 데이터를 포함한 데이터셋 로드
class MultiModalDataset(Dataset):
    def __init__(self, images_paths, features, labels, image_size=256):
        self.image_paths = images_paths
        self.features = torch.FloatTensor(features)  # 특징 데이터를 텐서로 변환
        self.labels = torch.LongTensor(labels)  # 레이블을 텐서로 변환
        self.image_size = image_size

        # 이미지 변환 파이프라인 정의
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # 이미지 텐서로 변환
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # 정규화
                                  std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 이미지 로드 및 전처리
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR -> RGB 변환
        image = cv2.resize(image, (self.image_size, self.image_size))  # 크기 조정
        image = self.transform(image)  # 변환 적용

        return {
            'image': image,  # 전처리된 이미지
            'features': self.features[idx],  # 특징 데이터
            'label': self.labels[idx]  # 레이블
        }

# Expert 네트워크 정의: 이미지와 특징 데이터를 처리
class Expert(nn.Module):
    def __init__(self, feature_dim, config):
        super().__init__()

        # CNN 블록 정의: 이미지 데이터를 인코딩
        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),  # Convolutional 레이어
                nn.BatchNorm2d(out_c, momentum=config.BN_MOMENTUM),  # Batch Normalization
                nn.ReLU(),  # 활성화 함수
                nn.Conv2d(out_c, out_c, 3, padding=1),  # 추가 Convolutional 레이어
                nn.BatchNorm2d(out_c, momentum=config.BN_MOMENTUM),  # Batch Normalization
                nn.ReLU(),
                nn.MaxPool2d(2),  # MaxPooling
                nn.Dropout2d(config.DROPOUT_RATE)  # 드롭아웃
            )

        # 이미지 인코더 구성
        self.image_encoder = nn.Sequential(
            conv_block(3, 64),
            conv_block(64, 128),
            conv_block(128, 256),
            conv_block(256, 512),
            nn.AdaptiveAvgPool2d((1, 1)),  # 평균 풀링
            nn.Flatten()  # 평탄화
        )

        # 특징 데이터 인코더 구성
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, config.HIDDEN_DIM * 4),  # 특징 차원 확장
            nn.BatchNorm1d(config.HIDDEN_DIM * 4, momentum=config.BN_MOMENTUM),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(config.HIDDEN_DIM * 4, config.HIDDEN_DIM * 2),
            nn.BatchNorm1d(config.HIDDEN_DIM * 2, momentum=config.BN_MOMENTUM),
            nn.ReLU(),
            nn.Linear(config.HIDDEN_DIM * 2, config.HIDDEN_DIM),
            nn.BatchNorm1d(config.HIDDEN_DIM, momentum=config.BN_MOMENTUM),
            nn.ReLU()
        )

        # 주의 메커니즘 정의: 이미지와 특징 데이터를 결합
        self.attention = nn.Sequential(
            nn.Linear(512 + config.HIDDEN_DIM, 256),
            nn.Tanh(),
            nn.Linear(256, 1),
            nn.Sigmoid()  # 주의 가중치 계산
        )

        # 최종 결합 레이어 구성
        self.combined = nn.Sequential(
            nn.Linear(512 + config.HIDDEN_DIM, config.HIDDEN_DIM * 4),
            nn.BatchNorm1d(config.HIDDEN_DIM * 4, momentum=config.BN_MOMENTUM),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(config.HIDDEN_DIM * 4, config.HIDDEN_DIM * 2),
            nn.BatchNorm1d(config.HIDDEN_DIM * 2, momentum=config.BN_MOMENTUM),
            nn.ReLU(),
            nn.Linear(config.HIDDEN_DIM * 2, 1)  # 최종 출력
        )

    def forward(self, images, features):
        img_feat = self.image_encoder(images)  # 이미지 특징 인코딩
        num_feat = self.feature_encoder(features)  # 숫자 특징 인코딩

        # 이미지와 특징 데이터 결합 및 주의 가중치 계산
        combined = torch.cat([img_feat, num_feat], dim=1)
        attention_weights = self.attention(combined)
        attended_features = combined * attention_weights

        # 최종 출력
        score = self.combined(attended_features)
        return score

# Focal Loss 정의: 불균형 클래스 문제를 해결하기 위한 손실 함수
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, class_weights=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = torch.tensor([1.0, 1.0, 3.0, 1.0]) if class_weights is None else class_weights

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets,
            weight=self.class_weights.to(inputs.device),
            label_smoothing=Config.LABEL_SMOOTHING,  # 레이블 스무딩 적용
            reduction='none'
        )
        pt = torch.exp(-ce_loss)  # 예측 확률
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss  # Focal Loss 계산
        return focal_loss.mean()
# EarlyStopping 클래스 정의: 학습 중 조기 종료를 위한 로직
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, path='best_model.pth'):
        self.patience = patience  # 개선되지 않는 에폭 수 허용치
        self.min_delta = min_delta  # 개선이라고 간주할 최소 변화량
        self.path = path  # 모델 저장 경로
        self.counter = 0  # 개선되지 않은 에폭 수
        self.best_loss = None  # 현재까지의 최저 손실값
        self.early_stop = False  # 조기 종료 플래그

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            # 첫 에폭에서 최적 손실값 초기화 및 모델 저장
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss > self.best_loss + self.min_delta:
            # 손실값이 개선되지 않을 경우 카운터 증가
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True  # 조기 종료 플래그 설정
        else:
            # 손실값이 개선되었을 경우 업데이트 및 카운터 리셋
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        # 모델의 상태 저장
        torch.save(model.state_dict(), self.path)

# MoEMultiModal 클래스 정의: Mixture of Experts (MoE) 기반 멀티모달 모델
class MoEMultiModal(nn.Module):
    def __init__(self, num_classes, feature_dim, config):
        super().__init__()
        self.num_classes = num_classes

        # 클래스별 전문가 네트워크 생성
        self.experts = nn.ModuleList([
            Expert(feature_dim, config) for _ in range(num_classes)
        ])

        # 게이트 네트워크 정의: 입력 특징에 따라 전문가 선택 가중치 계산
        self.gate = nn.Sequential(
            nn.Linear(feature_dim, config.HIDDEN_DIM * 2),
            nn.BatchNorm1d(config.HIDDEN_DIM * 2, momentum=config.BN_MOMENTUM),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(config.HIDDEN_DIM * 2, config.HIDDEN_DIM),
            nn.BatchNorm1d(config.HIDDEN_DIM, momentum=config.BN_MOMENTUM),
            nn.ReLU(),
            nn.Linear(config.HIDDEN_DIM, num_classes),
            nn.Softmax(dim=1)  # 클래스별 선택 확률 계산
        )

    def forward(self, images, features):
        # 게이트 네트워크에서 전문가 선택 가중치 계산
        gate_weights = self.gate(features)

        # 전문가 네트워크의 출력 계산
        expert_outputs = []
        for expert in self.experts:
            out = expert(images, features)
            expert_outputs.append(out)

        expert_outputs = torch.cat(expert_outputs, dim=1)  # 전문가 출력 결합
        final_output = expert_outputs * gate_weights  # 가중치 적용

        return final_output

# 한 에폭 동안 모델 평가 함수
def evaluate_one_epoch(model, data_loader, criterion, device):
    model.eval()  # 평가 모드로 전환
    total_loss = 0.0
    all_preds = []  # 예측값 저장
    all_labels = []  # 실제 레이블 저장

    with torch.no_grad():
        for batch in data_loader:
            images = batch['image'].to(device)  # 이미지 데이터 로드
            features = batch['features'].to(device)  # 특징 데이터 로드
            labels = batch['label'].to(device)  # 레이블 로드

            outputs = model(images, features)  # 모델 출력
            loss = criterion(outputs, labels)  # 손실 계산
            total_loss += loss.item()

            _, preds = torch.max(outputs, 1)  # 예측값 계산
            all_preds.extend(preds.cpu().numpy())  # 예측값 저장
            all_labels.extend(labels.cpu().numpy())  # 실제 레이블 저장

    avg_loss = total_loss / len(data_loader)  # 평균 손실값 계산
    accuracy = accuracy_score(all_labels, all_preds)  # 정확도 계산
    f1 = f1_score(all_labels, all_preds, average='macro')  # F1 점수 계산

    return avg_loss, accuracy, f1

# 학습률 조정을 위한 워밍업 스케줄 계산
def get_lr_multiplier(epoch, warmup_epochs):
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    return 1.0

# MoE 모델 학습 함수
def train_moe_model(model, train_loader, val_loader, criterion, optimizer, config, device):
    early_stopping = EarlyStopping(
        patience=config.EARLY_STOP_PATIENCE,
        path=os.path.join(config.SAVE_DIR, 'best_moe_model.pth')
    )

    # 학습률 스케줄러 설정
    if config.USE_SCHEDULER:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=config.SCHEDULER_PATIENCE,
            factor=config.SCHEDULER_FACTOR, min_lr=config.MIN_LR, verbose=True
        )

    # 학습 및 검증 성능 기록을 위한 리스트 초기화
    train_losses, train_accs, train_f1s = [], [], []
    val_losses, val_accs, val_f1s = [], [], []

    for epoch in range(config.NUM_EPOCHS):
        # 학습률 워밍업 적용
        if epoch < config.WARMUP_EPOCHS:
            lr_multiplier = get_lr_multiplier(epoch, config.WARMUP_EPOCHS)
            for param_group in optimizer.param_groups:
                param_group['lr'] = config.LEARNING_RATE * lr_multiplier

        # 학습 모드
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []

        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.NUM_EPOCHS}'):
            images = batch['image'].to(device)  # 이미지 데이터
            features = batch['features'].to(device)  # 특징 데이터
            labels = batch['label'].to(device)  # 레이블 데이터

            optimizer.zero_grad()  # 그래디언트 초기화
            outputs = model(images, features)  # 모델 출력
            loss = criterion(outputs, labels)  # 손실 계산

            # 게이트 희소성 규제 추가
            gate_sparsity = 0.0
            for expert in model.experts:
                gate_sparsity += torch.mean(torch.abs(expert.combined[-1].weight))

            total_loss = loss + config.SPARSITY_WEIGHT * gate_sparsity  # 총 손실 계산
            total_loss.backward()  # 역전파

            # 그래디언트 클리핑
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
            optimizer.step()  # 가중치 업데이트

            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        train_loss /= len(train_loader)
        train_acc = accuracy_score(train_labels, train_preds)
        train_f1 = f1_score(train_labels, train_preds, average='macro')

        # 검증 단계
        val_loss, val_acc, val_f1 = evaluate_one_epoch(model, val_loader, criterion, device)

        # 성능 기록
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        train_f1s.append(train_f1)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_f1s.append(val_f1)

        # 진행 상태 출력
        print(f'\nEpoch {epoch+1}/{config.NUM_EPOCHS}:')
        print(f'Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}')
        print(f'Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}')

        if config.USE_SCHEDULER:
            scheduler.step(val_loss)

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("\nEarly stopping triggered")
            break

    # 학습 곡선 저장
    plot_metrics(train_losses, val_losses, 'Loss', os.path.join(config.SAVE_DIR, 'loss_curves.png'))
    plot_metrics(train_accs, val_accs, 'Accuracy', os.path.join(config.SAVE_DIR, 'accuracy_curves.png'))
    plot_metrics(train_f1s, val_f1s, 'F1 Score', os.path.join(config.SAVE_DIR, 'f1_curves.png'))

# 혼동 행렬 시각화 함수
def plot_confusion_matrix(true_labels, pred_labels, class_names, save_path):
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# 학습 곡선 시각화 함수
def plot_metrics(train_metrics, val_metrics, metric_name, save_path):
    plt.figure(figsize=(10, 5))
    plt.plot(train_metrics, label=f'Train {metric_name}')
    plt.plot(val_metrics, label=f'Val {metric_name}')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.legend()
    plt.title(f'{metric_name} over Training')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# 클래스 가중치 계산 함수
def calculate_class_weights(y_train):
    class_counts = np.bincount(y_train)
    total = len(y_train)
    weights = total / (len(class_counts) * class_counts)
    return torch.FloatTensor(weights)

# 메인 함수
def main():
    torch.cuda.empty_cache()  # CUDA 메모리 초기화
    config = Config()  # Config 인스턴스 생성

    # 데이터 로드
    with open(os.path.join(FEATURE_DIR, 'multimodal_data.pkl'), 'rb') as f:
        data = pickle.load(f)

    # 클래스 가중치 계산
    class_weights = calculate_class_weights(data['y_train'])

    # 데이터셋 및 데이터로더 설정
    train_dataset = MultiModalDataset(
        data['train_paths'], data['X_train'], data['y_train'],
        image_size=config.IMAGE_SIZE
    )
    val_dataset = MultiModalDataset(
        data['val_paths'], data['X_val'], data['y_val'],
        image_size=config.IMAGE_SIZE
    )
    test_dataset = MultiModalDataset(
        data['test_paths'], data['X_test'], data['y_test'],
        image_size=config.IMAGE_SIZE
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE,
        shuffle=True, num_workers=config.NUM_WORKERS
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE,
        shuffle=False, num_workers=config.NUM_WORKERS
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.BATCH_SIZE,
        shuffle=False, num_workers=config.NUM_WORKERS
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MoEMultiModal(
        num_classes=len(data['label_encoder'].classes_),
        feature_dim=data['X_train'].shape[1],
        config=config
    ).to(device)

    criterion = FocalLoss(gamma=2, class_weights=class_weights)  # Focal Loss 설정
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )

    # 모델 학습
    print("Starting training...")
    train_moe_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        config=config,
        device=device
    )

    # 최적 모델 로드 및 최종 평가
    print("\nLoading best model for final evaluation...")
    model.load_state_dict(torch.load(os.path.join(config.SAVE_DIR, 'best_moe_model.pth')))

    print("\nValidation Set Performance:")
    val_loss, val_acc, val_f1 = evaluate_one_epoch(model, val_loader, criterion, device)
    print(f"Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}, F1 Score: {val_f1:.4f}")

    print("\nTest Set Performance:")
    test_loss, test_acc, test_f1 = evaluate_one_epoch(model, test_loader, criterion, device)
    print(f"Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}, F1 Score: {test_f1:.4f}")

    # 테스트 세트에서 세부 분류 결과 출력
    all_preds = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            features = batch['features'].to(device)
            labels = batch['label'].to(device)

            outputs = model(images, features)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("\nDetailed Classification Report:")
    print(classification_report(all_labels, all_preds,
                              target_names=data['label_encoder'].classes_))

    # 혼동 행렬 저장
    plot_confusion_matrix(
        all_labels, all_preds,
        data['label_encoder'].classes_,
        os.path.join(config.SAVE_DIR, 'confusion_matrix.png')
    )

# 스크립트 실행
if __name__ == "__main__":
    main()

