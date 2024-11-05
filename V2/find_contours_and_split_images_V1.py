import os
import cv2 as cv
import numpy as np
import sys

folder_path = './csv_to_images/'

def load_images_from_folder(folder_path):
    
    images = []
    labels = []

    for filename in os.listdir(folder_path):

        img_path = os.path.join(folder_path, filename)
      
        if os.path.isfile(img_path):

           img = cv.imread(img_path)
           img = cv.cvtColor(img,cv.COLOR_RGB2BGR) # 추가
           images.append(img)
           
           #cv.imshow("asdf",img)
           #cv.waitKey(0)

           _label = (filename.split('.')[0])
           _new = ""

           for i in range(6):
              _new += _label[i]
                             
        labels.append(_new) # 각 이미지에 대한 레이블 값을 [리스트 형태]로 묶음. ex) image 1 , [E,1,2,3,4,5]

    print("len",len(images))

    return images, labels



def split_6_imgs(image):

    #img_height, img_width, img_channel = image.shape
    
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    # 윤곽선을 찾습니다.
    contours, _ = cv.findContours(gray, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

    print("contours",len(contours))

    # 찾은 윤곽선을 이미지에 그립니다.
    cv.drawContours(image, contours, -1, (0, 255, 255), 2)

    # 결과 이미지를 출력합니다.
    #cv.imshow('Contours', image)
    #cv.waitKey(0)
    
    
    for contour in contours:
       
       area = cv.contourArea(contour)
       x, y, w, h = cv.boundingRect(contour)
       if area >= 9000.0 and area <= 690000.0 and w/h >= 0.30:
        print("area",area)
        print("w/h",w/h)
        cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)  # 초록색으로 Bounding Box 표시, 두께는 2
        cv.imshow("asdf",image)
        cv.waitKey()

    return image  # Bounding Box가 표시된 이미지를 반환


# 사용 예제
folder_path = './csv_to_images'
images,labels = load_images_from_folder(folder_path)


for image in images:
  split_6_imgs(image)

