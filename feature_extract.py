import numpy as np
import pandas as pd
import cv2
import os
from tqdm import tqdm
import pickle
from sklearn.preprocessing import LabelEncoder

# 경로 설정
PREPROCESS_DIR = 'data/preprocess_data'
FEATURE_DIR = 'data/feature_data'
os.makedirs(FEATURE_DIR, exist_ok=True)

class FeatureExtractor:
    def __init__(self, input_size=256):
        self.input_size = input_size
        self.label_encoder = LabelEncoder()

    def extract_color_features(self, img):
        """컬러 관련 특징 추출"""
        # 각 채널별 평균과 표준편차
        means = img.mean(axis=(0, 1))
        stds = img.std(axis=(0, 1))

        # 각 채널별 히스토그램
        features = []
        for i in range(3):  # RGB 각 채널
            hist = cv2.calcHist([img], [i], None, [32], [0, 256])
            hist = hist.flatten() / hist.sum()  # 정규화
            features.extend(hist)

        features.extend(means)
        features.extend(stds)
        return np.array(features)

    def extract_texture_features(self, img):
        """텍스처 특징 추출"""
        # 그레이스케일 변환
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Sobel 엣지 검출
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        # 엣지 강도와 방향
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        direction = np.arctan2(sobely, sobelx)

        # 통계적 특징
        features = [
            np.mean(magnitude),  # 평균 엣지 강도
            np.std(magnitude),   # 엣지 강도의 표준편차
            np.percentile(magnitude, 90),  # 90번째 퍼센타일
            np.mean(direction),  # 평균 엣지 방향
            np.std(direction),   # 엣지 방향의 표준편차
        ]

        # LBP와 유사한 local 패턴 분석
        kernel_size = 3
        local_mean = cv2.blur(gray, (kernel_size, kernel_size))
        pattern = (gray > local_mean).astype(np.uint8)
        pattern_hist = cv2.calcHist([pattern], [0], None, [2], [0, 2])
        pattern_hist = pattern_hist.flatten() / pattern_hist.sum()

        features.extend(pattern_hist)
        return np.array(features)

    def extract_shape_features(self, img):
        """형태 관련 특징 추출"""
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # 이진화 (이미 엣지 정보가 포함된 이미지이므로)
        _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

        # 컨투어 찾기
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return np.zeros(5)

        # 가장 큰 컨투어 선택
        largest_contour = max(contours, key=cv2.contourArea)

        # 특징 추출
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        hull = cv2.convexHull(largest_contour)
        hull_area = cv2.contourArea(hull)

        features = [
            area / (self.input_size ** 2),  # 정규화된 면적
            perimeter / (4 * self.input_size),  # 정규화된 둘레
            hull_area / (self.input_size ** 2),  # 정규화된 컨벡스 헐 면적
            4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0,  # 원형도
            area / hull_area if hull_area > 0 else 0  # 볼록도
        ]

        return np.array(features)

    def process_dataset(self, split='train'):
        """데이터셋 처리 및 특징 추출"""
        data_dir = os.path.join(PREPROCESS_DIR, split)
        features_list = []
        labels = []
        image_paths = []

        for class_name in os.listdir(data_dir):
            class_dir = os.path.join(data_dir, class_name)
            if not os.path.isdir(class_dir):
                continue

            print(f"Processing {split} - {class_name}")
            for img_name in tqdm(os.listdir(class_dir)):
                if not img_name.endswith('.jpg'):
                    continue

                img_path = os.path.join(class_dir, img_name)
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # 특징 추출
                color_features = self.extract_color_features(img)
                texture_features = self.extract_texture_features(img)
                shape_features = self.extract_shape_features(img)

                # 모든 특징 합치기
                all_features = np.concatenate([
                    color_features,
                    texture_features,
                    shape_features
                ])

                features_list.append(all_features)
                labels.append(class_name)
                image_paths.append(img_path)

        # 특징을 numpy 배열로 변환
        X = np.array(features_list)

        # 레이블 인코딩 (train일 때만 fit)
        if split == 'train':
            y = self.label_encoder.fit_transform(labels)
        else:
            y = self.label_encoder.transform(labels)

        return X, y, image_paths

    def prepare_data(self):
        """전체 데이터셋 준비"""
        # 이미 분할된 데이터셋 처리
        print("Processing training data...")
        X_train, y_train, train_paths = self.process_dataset('train')

        print("Processing validation data...")
        X_val, y_val, val_paths = self.process_dataset('val')

        print("Processing test data...")
        X_test, y_test, test_paths = self.process_dataset('test')

        # 데이터 저장
        data = {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'train_paths': train_paths,
            'val_paths': val_paths,
            'test_paths': test_paths,
            'label_encoder': self.label_encoder
        }

        with open(os.path.join(FEATURE_DIR, 'multimodal_data.pkl'), 'wb') as f:
            pickle.dump(data, f)

        print("\nFeature shapes:")
        print(f"X_train: {X_train.shape}")
        print(f"X_val: {X_val.shape}")
        print(f"X_test: {X_test.shape}")

        print("\nSample counts:")
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Test samples: {len(X_test)}")

        return data

# 특징 추출 실행
if __name__ == "__main__":
    extractor = FeatureExtractor()
    data = extractor.prepare_data()
    print("\nFeature extraction completed and saved!")