import os
import glob
import numpy as np
import pandas as pd

from scipy.ndimage import convolve
from PIL import Image


def convolution_average(data,kernel_size):

    # 커널 생성
    kernel = np.ones((kernel_size,kernel_size), dtype=np.float64)

    # nan 값을 제외한 평균 계산을 위해 nan이 아닌 값을 1로 설정한 마스크 배열 생성
    mask = ~np.isnan(data)
    masked_data = np.nan_to_num(data)  # nan 값을 0으로 변환한 데이터 배열

    # 마스크 배열에 커널을 적용하여 유효한 요소의 수 계산
    valid_count = convolve(mask.astype(np.float64), kernel, mode='constant', cval=0.0)

    # 마스크를 적용한 데이터 배열에 커널을 적용하여 합 계산
    summed_data = convolve(masked_data, kernel, mode='constant', cval=0.0)

    # 유효한 요소 수로 합을 나누어 평균 계산
    averaged_data = np.divide(summed_data, valid_count, where=valid_count != 0)

    return averaged_data

def count_cells_in_range(data, kernel_size, min_val=4, max_val=8):
    
    # 커널 생성
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float64)
    
    # 유효 범위 내의 셀을 1로 설정하는 마스크 생성
    range_mask = (data >= min_val) & (data <= max_val)
    
    # nan이 아닌 값을 1로 설정한 마스크 배열 생성
    valid_mask = ~np.isnan(data)
    
    # 범위 마스크와 유효 마스크를 결합
    combined_mask = range_mask & valid_mask
    
    # 유효 범위 내의 셀의 개수 계산
    counted_cells = convolve(combined_mask.astype(np.float64), kernel, mode='constant', cval=0.0)
    
    return counted_cells

start_row, start_col = 60,157
end_row, end_col = 1350,700


input_folder_path = './csv'
output_folder_path = './csv_to_images'
csv_files = glob.glob(os.path.join(input_folder_path, '*csv'))

# 51x51 커널 크기 정의
kernel_size = 31

# 각 CSV 파일 처리
for csv_file in csv_files:

    print(f"파일 처리 중: {csv_file}")

    # CSV 파일 읽기
    df = pd.read_csv(csv_file, low_memory=False)

    # 지정된 범위로 데이터프레임 자르기
    cropped_df = df.iloc[start_row:end_row, start_col:end_col]
    
    # numpy 배열로 변환
    data = cropped_df.to_numpy(dtype=np.float64)

    # 평균 계산
    averaged_data = convolution_average(data, kernel_size)

    # 결과를 데이터프레임으로 변환
    averaged_df = pd.DataFrame(averaged_data, index=cropped_df.index, columns=cropped_df.columns)

    # 범위 내의 셀 개수 계산
    cells_in_range_count = count_cells_in_range(data, kernel_size)
    
    cells_in_range_df = pd.DataFrame(cells_in_range_count, index=cropped_df.index, columns=cropped_df.columns)

    # 이미지를 생성합니다.
    img_height, img_width = averaged_df.shape
    #image = Image.new('RGB', (img_width, img_height), "white")  # 흰색 배경의 이미지 생성
    #pixels = image.load()
    image_array = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255  # 흰색 배경의 이미지 생성

    # 각 셀의 값을 기준으로 픽셀 색상을 설정합니다.
    mean = averaged_data
    count_4_to_8 = cells_in_range_count
    ratio_4_to_8 = count_4_to_8 / (kernel_size * kernel_size)
    #print("max",np.max(ratio_4_to_8))
    
     
    # 조건에 맞는 위치에 대해 색상을 검은색으로 설정
    # mask = (mean > 4) & (mean < 12) & (ratio_4_to_8 > 0.040) & (ratio_4_to_8 < 0.39)    
    # mask = (mean > 4) & (mean < 12) & (ratio_4_to_8 > 0.032) & (ratio_4_to_8 < 0.45) #& ( ((data > 4) & (data < 9)) )

    #mask = (mean > 3) & (mean < 12) & (ratio_4_to_8 > 0.010) & (ratio_4_to_8 <= 1.0) #& ( ((data > 4) & (data < 9)) ) #정상 중 제대로 잘리는 부분
    mask = (mean > 3) & (mean < 14) & (ratio_4_to_8 > 0.032) & (ratio_4_to_8 <= 1.0) #& ( ((data > 4) & (data < 9)) ) #정상 중 제대로 안 잘리는 부분
    
    image_array[mask] = [0, 0, 0]  # 검은색으로 설정

    # NumPy 배열을 PIL 이미지로 변환
    image = Image.fromarray(image_array, 'RGB')
    

    # 이미지를 90도 오른쪽으로 회전
    image_rotated = image.rotate(-90, expand=True)  # `expand=True`는 회전 후 이미지 크기를 조절합니다.

    # 회전된 이미지를 저장합니다.
    output_file_path = os.path.join(output_folder_path, os.path.basename(csv_file).replace('.csv', '.png'))   
    image_rotated.save(output_file_path)


