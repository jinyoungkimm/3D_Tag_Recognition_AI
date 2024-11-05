import os
import cv2
import numpy as np
from imgaug import augmenters as iaa


# 이미지가 저장된 폴더 경로
#folder_path = './for_training_images/alphabat_roi_images_all/'
#output_folder = './for_training_images/augmented_alphabat_roi_images'
folder_path = './for_training_images/numeric_roi_images_all'
output_folder = './for_training_images/augmented_numeric_roi_images'

# 이미지 파일 확장자
image_extensions = ['.png']

# 증강할 이미지 수
augmentation_factor = 5

# 증강할 이미지를 저장할 폴더

os.makedirs(output_folder, exist_ok=True)

# 증강 기법 정의
seq = iaa.Sequential([
    iaa.Fliplr(0.5),  # 좌우 반전
    iaa.Flipud(0.5),  # 상하 반전
    #iaa.Affine(rotate=(-20, 20)),  # 회전
    iaa.GaussianBlur(sigma=(0, 3.0)),  # 가우시안 블러
    iaa.AdditiveGaussianNoise(scale=(0, 0.1 * 255)),  # 가우시안 노이즈
    iaa.Multiply((0.8, 1.2)),  # 밝기 조절
    iaa.Affine(shear=(-16, 16))  # 전단 변환
])

# 폴더 내의 각 이미지 파일에 대해 증강 수행
for filename in os.listdir(folder_path):
    if any(ext in filename.lower() for ext in image_extensions):

        image_path = os.path.join(folder_path, filename)
        img = cv2.imread(image_path)

        # 이미지 증강
        augmented_images = [img for _ in range(augmentation_factor)]
        augmented_images = seq.augment_images(augmented_images)

        # 증강된 이미지를 저장
        for idx, augmented_img in enumerate(augmented_images):

            output_filename = f"{os.path.splitext(filename)[0]}_augmented_{idx}.png"
            output_path = os.path.join(output_folder, output_filename)
            #augmented_img = cv2.resize(augmented_img,(28,28))
            cv2.imwrite(output_path, augmented_img)