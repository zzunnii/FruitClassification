import cv2
import numpy as np
import random
from scipy.ndimage import gaussian_filter, map_coordinates
import os

def augment_with_noise(image):
    """이미지에 랜덤 노이즈 추가"""
    noise = np.random.normal(0, random.uniform(5, 20), image.shape).astype(np.uint8)
    noisy_img = cv2.add(image, noise)
    return noisy_img

def basic_geometric_augment(image):
    """샘플 수 맞추기용 기본 지오메트릭 변환"""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    transforms = [
        # 회전 (최소한의 각도)
        lambda img: cv2.warpAffine(img, cv2.getRotationMatrix2D(center, np.random.uniform(-10, 10), 1.0), (w, h)),
        # 스케일 (작은 변화)
        lambda img: cv2.resize(img, None, fx=np.random.uniform(0.9, 1.1), fy=np.random.uniform(0.9, 1.1)),
        # 밝기 (최소 조정)
        lambda img: cv2.convertScaleAbs(img, alpha=np.random.uniform(0.9, 1.1), beta=np.random.randint(-10, 10))
    ]
    return random.choice(transforms)(image)

def apply_elastic_transform(image, alpha, sigma):
    """안전한 탄성 변환 적용"""
    try:
        # 각 채널별로 처리
        result = np.zeros_like(image)
        random_state = np.random.RandomState(None)

        # 변위 필드 생성 (한 번만)
        shape = image.shape[:2]
        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha

        # 격자 생성
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))

        # 좌표 범위 제한
        dx = np.clip(dx, -shape[0]//4, shape[0]//4)
        dy = np.clip(dy, -shape[1]//4, shape[1]//4)

        indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))

        # 각 채널에 대해 변환 적용
        for i in range(3):
            result[..., i] = map_coordinates(image[..., i], indices, order=1).reshape(shape)

        # 결과 값 범위 제한
        result = np.clip(result, 0, 255)

        return result.astype(np.uint8)

    except Exception as e:
        print(f"Error in elastic transform: {e}")
        return image  # 오류 발생 시 원본 이미지 반환

def simulate_overlap(image):
    """과일 겹침 시뮬레이션"""
    h, w = image.shape[:2]
    overlay = np.zeros_like(image)

    # 랜덤 위치에 반투명한 원형 영역 생성
    center = (np.random.randint(w//4, 3*w//4), np.random.randint(h//4, 3*h//4))
    radius = np.random.randint(w//6, w//3)

    cv2.circle(overlay, center, radius, (255, 255, 255), -1)
    alpha = np.random.uniform(0.3, 0.7)

    return cv2.addWeighted(image, 1, overlay, alpha, 0)

def apply_partial_occlusion(image):
    """부분 가림 효과 적용"""
    h, w = image.shape[:2]
    mask = np.ones_like(image)

    # 랜덤한 사각형 영역 생성
    x1, y1 = np.random.randint(0, w//2), np.random.randint(0, h//2)
    x2, y2 = x1 + np.random.randint(w//4, w//2), y1 + np.random.randint(h//4, h//2)

    mask[y1:y2, x1:x2] = 0
    return image * mask

def apply_perspective_transform(image):
    """원근 변환 적용"""
    h, w = image.shape[:2]

    # 소스 포인트
    src_points = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

    # 목표 포인트 (약간의 랜덤 변형)
    dst_points = np.float32([
        [np.random.randint(0, w//8), np.random.randint(0, h//8)],
        [w - np.random.randint(0, w//8), np.random.randint(0, h//8)],
        [w - np.random.randint(0, w//8), h - np.random.randint(0, h//8)],
        [np.random.randint(0, w//8), h - np.random.randint(0, h//8)]
    ])

    # 변환 행렬 계산 및 적용
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    return cv2.warpPerspective(image, matrix, (w, h))
