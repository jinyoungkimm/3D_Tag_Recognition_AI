import os
import glob 
import numpy as np
import pandas as pd
from scipy.ndimage import generic_filter
from PIL import Image

start_row , start_col = 60,157
end_row, end_col = 1350, 700

input_folder_path = './csv'

csv_files = glob.glob(os.path.join(input_folder_path,'*csv'))

# 51x51 커널 크기 정의
kernel_size = 51

# NaN을 제외하고 각 셀에 대해 컨볼루션 및 평균을 적용하는 함수
def apply_convolution_and_mean(data):

    def nanmean_filter(values):
        
        valid_values = values[~np.isnan(values)]

        if len(valid_values) > 0:
            return np.mean(valid_values)

        else:
            return np.nan
    
    footprint = np.ones((kernel_size, kernel_size))
    convolved = generic_filter(data, nanmean_filter, footprint=footprint, mode='constant', cval=np.nan)

    return convolved

# 각 셀에 대해 컨볼루션을 적용하고 12보다 큰 값의 개수를 계산하는 함수
def apply_convolution_and_count(data):

    def count_greater_than_12(values):

        #valid_values = values[~np.isnan(values)]
        
        #return np.sum( valid_values > 12)
        
        valid_values = values[~np.isnan(values)]
        
        # 조건에 맞는 값의 수를 셉니다
        count = np.sum((valid_values >= 4) & (valid_values <= 8))
        
        return count
    
    footprint = np.ones((kernel_size, kernel_size))
    count_result = generic_filter(data, count_greater_than_12, footprint=footprint, mode='constant', cval=np.nan)
    
    return count_result

# 각 CSV 파일 처리
for csv_file in csv_files:

    print(f"파일 처리 중: {csv_file}")

    # CSV 파일 읽기
    df = pd.read_csv(csv_file, low_memory=False)

    # 지정된 범위로 데이터프레임 자르기
    cropped_df = df.iloc[start_row:end_row, start_col:end_col] #FB62

    # 컨볼루션을 위해 잘린 데이터프레임을 numpy 배열로 변환
    cropped_array = cropped_df.to_numpy()

    # NaN을 제외하고 컨볼루션 및 평균 계산 적용
    mean_convolution_result= apply_convolution_and_mean(cropped_array)
    
    # 12보다 큰 값의 개수를 계산하는 컨볼루션 적용
    count_greater_than_12_result = apply_convolution_and_count(cropped_array)

    # 결과를 다시 데이터프레임으로 변환
    mean_convolution_df = pd.DataFrame(mean_convolution_result, index=cropped_df.index, columns=cropped_df.columns)
    # 결과를 다시 데이터프레임으로 변환
    count_greater_than_12_df = pd.DataFrame(count_greater_than_12_result, index=cropped_df.index, columns=cropped_df.columns)


    # 결과 데이터프레임 출력
    print("평균 컨볼루션 결과:")
    print(mean_convolution_df)


    # 결과 데이터프레임 출력
    print("12보다 큰 값의 개수 결과:")
    print(count_greater_than_12_df)


    # 이미지를 생성합니다.
    img_height, img_width = mean_convolution_df.shape
    image = Image.new('RGB', (img_width, img_height), "white")  # 흰색 배경의 이미지 생성
    pixels = image.load()

    # 각 셀의 값을 기준으로 픽셀 색상을 설정합니다.
    for i in range(img_height):
        for j in range(img_width):
            
            cell_value = cropped_df.iat[i,j]
            
            mean = mean_convolution_df.iat[i, j]
            count_12 = count_greater_than_12_df.iat[i,j]
            ratio_12 = count_12/(kernel_size*kernel_size)

            # ( (5 < cell_value < 8) ) and (5 < mean < 11.5) and (ratio_12 < 0.5):
            if  (5 < mean < 11.5) and (0.05 < ratio_12 < 0.23):
                pixels[j, i] = (0, 0, 0)  # 검은색으로 설정
    print("max",max)
    # 이미지를 저장합니다.
    image_path = os.path.join(input_folder_path, f"output_{os.path.basename(csv_file).split('.')[0]}.png")
    image.save(image_path)
    print(f"이미지 저장됨: {image_path}")




     



