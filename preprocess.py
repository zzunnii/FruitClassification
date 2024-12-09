from augment_def import simulate_overlap, apply_partial_occlusion, apply_perspective_transform, apply_elastic_transform, \
    basic_geometric_augment
# 필요한 라이브러리 임포트
from scipy.ndimage import gaussian_filter, map_coordinates
import os, cv2, numpy as np
from pathlib import Path
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split

# 기본 설정
TRAIN_DIR = 'data/train'  # 원본 학습 데이터 경로
TEST_DIR = 'data/test'  # 원본 테스트 데이터 경로
PREPROCESS_DIR = 'data/preprocess_data'  # 전처리된 데이터 저장 경로
TARGET_SIZE = 256  # 이미지 크기 표준화 목표 크기

random.seed(42)  # 재현성을 위한 랜덤 시드 설정


def create_directories():
    """전처리된 데이터를 저장할 디렉토리 구조 생성

    - 기존 전처리 디렉토리가 있다면 삭제
    - train, validation, test 세 가지 분할을 위한 디렉토리 생성
    """
    if os.path.exists(PREPROCESS_DIR):
        shutil.rmtree(PREPROCESS_DIR)
    os.makedirs(os.path.join(PREPROCESS_DIR, 'train'), exist_ok=True)
    os.makedirs(os.path.join(PREPROCESS_DIR, 'val'), exist_ok=True)
    os.makedirs(os.path.join(PREPROCESS_DIR, 'test'), exist_ok=True)


def remove_background(image):
    """이미지에서 과일 영역을 추출하고 배경 제거

    Args:
        image: 입력 BGR 이미지

    Returns:
        배경이 제거된 이미지

    Note:
        1. HSV 색공간에서 과일 색상 범위에 해당하는 마스크 생성
        2. 엣지 검출을 통한 윤곽선 마스크 생성
        3. 두 마스크를 결합하고 모폴로지 연산으로 정제
        4. 최종 마스크를 이용해 원본에서 과일 영역만 추출
    """
    # BGR에서 HSV로 변환
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 과일 색상 범위에 대한 마스크들
    masks = [
        cv2.inRange(hsv, np.array([0, 20, 20]), np.array([15, 255, 255])),  # 빨강1
        cv2.inRange(hsv, np.array([165, 20, 20]), np.array([180, 255, 255])),  # 빨강2
        cv2.inRange(hsv, np.array([15, 20, 20]), np.array([40, 255, 255])),  # 노랑
        cv2.inRange(hsv, np.array([5, 20, 20]), np.array([25, 255, 255]))  # 주황
    ]

    # 모든 색상 마스크 통합
    color_mask = masks[0]
    for mask in masks[1:]:
        color_mask = cv2.bitwise_or(color_mask, mask)

    # 엣지 검출을 통한 윤곽선 마스크 생성
    edges = cv2.Canny(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 100, 200)
    final_mask = cv2.bitwise_or(color_mask, edges)

    # 모폴로지 연산으로 마스크 정제
    kernel = np.ones((5, 5), np.uint8)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)  # 작은 구멍 제거
    final_mask = cv2.dilate(final_mask, kernel, iterations=1)  # 마스크 영역 확장

    # 마스크 적용하여 배경 제거
    return cv2.bitwise_and(image, image, mask=final_mask)


def center_on_black(image):
    """이미지를 검은 배경 중앙에 배치하고 크기 표준화

    Args:
        image: 입력 이미지

    Returns:
        표준화된 크기(TARGET_SIZE x TARGET_SIZE)의 검은 배경 중앙에 배치된 이미지

    Note:
        1. TARGET_SIZE x TARGET_SIZE 크기의 검은 배경 생성
        2. 입력 이미지의 비율을 유지하면서 크기 조정 (80% 스케일)
        3. 조정된 이미지를 검은 배경 중앙에 배치
    """
    # 검은 배경 생성
    black_bg = np.zeros((TARGET_SIZE, TARGET_SIZE, 3), dtype=np.uint8)

    # 입력 이미지 크기
    h, w = image.shape[:2]

    # 비율을 유지하면서 크기 조정 (80% 스케일)
    scale = min(TARGET_SIZE / h, TARGET_SIZE / w) * 0.8
    new_h, new_w = int(h * scale), int(w * scale)

    # 고품질 리사이즈 수행
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    # 중앙 배치를 위한 오프셋 계산
    y_offset = (TARGET_SIZE - new_h) // 2
    x_offset = (TARGET_SIZE - new_w) // 2

    # 리사이즈된 이미지를 검은 배경 중앙에 배치
    black_bg[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    return black_bg


def split_train_val_files(input_dir):
    """학습 데이터를 학습/검증 세트로 분할

    Args:
        input_dir: 원본 학습 데이터 디렉토리 경로

    Returns:
        train_files: 학습용 파일 리스트
        val_files: 검증용 파일 리스트

    Note:
        - 클래스별로 파일을 그룹화하여 분할 (클래스 밸런스 유지)
        - 각 클래스에서 80%는 학습용, 20%는 검증용으로 분할
        - 재현성을 위해 random_state=42 사용
    """
    class_files = {}
    # 클래스별로 파일 분류
    for filename in os.listdir(input_dir):
        if filename.endswith('.jpg'):
            class_name = filename.split('_')[0]  # 파일명의 첫 부분이 클래스명
            if class_name not in class_files:
                class_files[class_name] = []
            class_files[class_name].append(filename)

    train_files = []
    val_files = []

    # 클래스별로 train/val 분할
    for class_name, files in class_files.items():
        files.sort()  # 파일 순서 보장
        train_class_files, val_class_files = train_test_split(
            files, test_size=0.2, random_state=42, shuffle=True
        )
        train_files.extend(train_class_files)
        val_files.extend(val_class_files)

    return train_files, val_files

def preprocess_test_data():
    """테스트 데이터 전처리"""
    output_dir = os.path.join(PREPROCESS_DIR, 'test')
    os.makedirs(output_dir, exist_ok=True)

    for filename in tqdm(os.listdir(TEST_DIR), desc="Processing test data"):
        if not filename.endswith('.jpg'):
            continue

        class_name = filename.split('_')[0]
        class_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)

        image_path = os.path.join(TEST_DIR, filename)
        image = cv2.imread(image_path)
        if image is None:
            continue

        processed = remove_background(image)
        processed = center_on_black(processed)
        cv2.imwrite(os.path.join(class_dir, filename), processed)

def preprocess_and_select(files, split='train'):
    """데이터 전처리 및 기본 증강 수행

    Args:
        files: 처리할 파일 리스트
        split: 데이터 분할 종류 ('train', 'val', 'test')

    Note:
        1. 클래스별로 파일 분류
        2. 각 이미지에 대해:
           - 배경 제거
           - 크기 표준화 및 중앙 정렬
        3. 학습 데이터의 경우:
           - 클래스별 최대 샘플 수 계산
           - 부족한 클래스는 기본 증강으로 보충
    """
    output_dir = os.path.join(PREPROCESS_DIR, split)
    os.makedirs(output_dir, exist_ok=True)

    # 클래스별 파일 분류
    class_files = {}
    for filename in files:
        class_name = filename.split('_')[0]
        if class_name not in class_files:
            class_files[class_name] = []
        class_files[class_name].append(filename)

    # 학습 데이터의 경우 최대 샘플 수 계산
    if split == 'train':
        max_samples = max(len(files) for files in class_files.values())
        print(f"Maximum samples per class in {split}: {max_samples}")

    # 클래스별 처리
    for class_name, class_files_list in class_files.items():
        class_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)

        processed_images = []
        # 각 이미지 전처리
        for filename in tqdm(class_files_list, desc=f"Preprocessing {class_name} ({split})", leave=False):
            image_path = os.path.join(TRAIN_DIR, filename)
            image = cv2.imread(image_path)
            if image is None:
                continue
            # 배경 제거 및 표준화
            processed = remove_background(image)
            processed = center_on_black(processed)
            processed_images.append(processed)

        # 학습 데이터의 경우 샘플 수 보충
        if split == 'train':
            current_count = len(processed_images)
            if current_count < max_samples:
                needed = max_samples - current_count
                # 기본 증강으로 부족한 샘플 보충
                for _ in range(needed):
                    base_img = random.choice(processed_images)
                    processed_images.append(basic_geometric_augment(base_img))

        # 처리된 이미지 저장
        for i, img in enumerate(processed_images):
            save_name = f"{class_name}_{i + 1}.jpg"
            cv2.imwrite(os.path.join(class_dir, save_name), img)


def final_augment_image(image, class_name=None):
    """고급 증강 기법 적용

    Args:
        image: 입력 이미지
        class_name: 클래스 이름 (mixed 클래스 특화 증강을 위해 필요)

    Returns:
        augmented_results: (변환명, 변환된 이미지) 튜플의 리스트

    Note:
        세 가지 종류의 변환 적용:
        1. 기본 변환 (회전, 밝기, 대비)
        2. 복합 변환 (회전-스케일, 밝기-블러, 탄성)
        3. Mixed 클래스 특화 변환 (겹침, 가림, 원근)
    """
    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    # 1. 기본 변환 정의 (더 강화된 파라미터)
    basic_transforms = {
        'rotation': lambda img: cv2.warpAffine(
            img,
            cv2.getRotationMatrix2D(center, np.random.randint(-45, 45), 1.0),
            (w, h)
        ),
        'brightness': lambda img: cv2.convertScaleAbs(
            img,
            alpha=np.random.uniform(0.6, 1.4),
            beta=np.random.randint(-50, 50)
        ),
        'contrast': lambda img: cv2.convertScaleAbs(
            img,
            alpha=np.random.uniform(0.5, 1.5)
        )
    }

    # 2. 복합 변환 개선
    composite_transforms = {
        'rotation_scale': lambda img: cv2.resize(
            cv2.warpAffine(
                img,
                cv2.getRotationMatrix2D(
                    (img.shape[1] // 2, img.shape[0] // 2),
                    np.random.uniform(-45, 45),
                    np.random.uniform(0.8, 1.2)
                ),
                (img.shape[1], img.shape[0])
            ),
            None,
            fx=np.random.uniform(0.9, 1.1),
            fy=np.random.uniform(0.9, 1.1)
        ),
        'brightness_blur': lambda img: cv2.GaussianBlur(
            cv2.convertScaleAbs(
                img,
                alpha=np.random.uniform(0.6, 1.4),
                beta=np.random.randint(-30, 30)
            ),
            (5, 5),
            sigmaX=np.random.uniform(0.5, 1.5)
        ),
        'elastic_transform': lambda img: apply_elastic_transform(
            img,
            alpha=np.random.uniform(20, 50),
            sigma=np.random.uniform(5, 10)
        ),
    }

    # 3. Mixed 클래스 특화 변환
    mixed_specific_transforms = {
        'overlap_simulation': lambda img: simulate_overlap(img),
        'partial_occlusion': lambda img: apply_partial_occlusion(img),
        'perspective_transform': lambda img: apply_perspective_transform(img)
    }

    augmented_results = []

    # 기본 변환 적용
    for key in basic_transforms.keys():
        aug_img = basic_transforms[key](image)
        augmented_results.append((f'basic_{key}', aug_img))

    # 복합 변환 적용
    for key in composite_transforms.keys():
        aug_img = composite_transforms[key](image)
        augmented_results.append((f'composite_{key}', aug_img))

    # Mixed 클래스 특화 변환 적용
    if class_name == 'mixed':
        for key in mixed_specific_transforms.keys():
            aug_img = mixed_specific_transforms[key](image)
            augmented_results.append((f'mixed_{key}', aug_img))

    return augmented_results


def final_augmentation(split='train'):
    """최종 증강 (학습 데이터만)"""
    if split != 'train':
        return

    train_dir = os.path.join(PREPROCESS_DIR, 'train')
    for class_name in os.listdir(train_dir):
        class_path = os.path.join(train_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        # 원본 이미지만 선택 (class_숫자.jpg 형태)
        images = [f for f in os.listdir(class_path) if f.endswith('.jpg') and len(f.split('_')) == 2]
        images.sort()

        for filename in tqdm(images, desc=f"Final Augmenting {class_name}", leave=False):
            img_path = os.path.join(class_path, filename)
            img = cv2.imread(img_path)
            if img is None:
                continue

            aug_imgs = final_augment_image(img, class_name)
            base_name = os.path.splitext(filename)[0]

            for (transform_name, aimg) in aug_imgs:
                aug_path = os.path.join(class_path, f"{base_name}_aug_{transform_name}.jpg")
                cv2.imwrite(aug_path, aimg)


def plot_examples_for_mixed():
    class_dir = os.path.join(PREPROCESS_DIR, 'train', 'mixed')
    if not os.path.exists(class_dir):
        print("No mixed class directory found.")
        return
    images = [f for f in os.listdir(class_dir) if f.endswith('.jpg') and '_aug' not in f]
    if not images:
        print("No images found in mixed class.")
        return
    img_path = os.path.join(class_dir, images[0])
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    aug_imgs = final_augment_image(img)

    plt.figure(figsize=(15, 3))
    plt.subplot(1, len(aug_imgs) + 1, 1)
    plt.imshow(img_rgb)
    plt.title("Original")
    plt.axis('off')

    for i, (name, aimg) in enumerate(aug_imgs, 1):
        plt.subplot(1, len(aug_imgs) + 1, i + 1)
        plt.imshow(cv2.cvtColor(aimg, cv2.COLOR_BGR2RGB))
        plt.title(name)
        plt.axis('off')
    plt.show()


if __name__ == "__main__":
    create_directories()

    print("\nSplitting training data into train/val...")
    train_files, val_files = split_train_val_files(TRAIN_DIR)

    print("\nPreprocessing and augmenting training data...")
    preprocess_and_select(train_files, split='train')

    print("\nPreprocessing validation data...")
    preprocess_and_select(val_files, split='val')

    print("\nPreprocessing test data...")
    preprocess_test_data()

    print("\nApplying final augmentation to training data...")
    final_augmentation(split='train')

    # 최종 데이터 수 출력
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(PREPROCESS_DIR, split)
        for class_name in os.listdir(split_dir):
            class_path = os.path.join(split_dir, class_name)
            if os.path.isdir(class_path):
                count = len([f for f in os.listdir(class_path) if f.endswith('.jpg')])
                print(f"{split.capitalize()} - Class {class_name}: {count} images")

    print("\nShowing examples for mixed class final augmentation...")
    plot_examples_for_mixed()