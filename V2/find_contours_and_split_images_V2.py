import os
import cv2 as cv
import numpy as np
import sys

folder_path = './csv_to_images/'
output_folder = './cropped_images/'
#output_folder_alphatbat = './for_training_images/alphabat_roi_images/'
#output_folder_numeric = './for_training_images/numeric_roi_images/'

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
           _new = _label
           #_new = ""

           #for i in range(6):
           #   _new += _label[i]
                             
        labels.append(_new) # 각 이미지에 대한 레이블 값을 [리스트 형태]로 묶음. ex) image 1 , [E,1,2,3,4,5]
    
    print("len",len(images))

    return images, labels



def split_6_imgs_and_save(image,label):

    #img_height, img_width, img_channel = image.shape
    
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    # 윤곽선을 찾습니다.
    contours, _ = cv.findContours(gray, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

    # 찾은 윤곽선을 이미지에 그립니다.
    #cv.drawContours(image, contours, -1, (0, 255, 255), 2)
    
    # 결과 이미지를 출력합니다
    #cv.imshow('Contours', image)
    #cv.waitKey(0)
    
    BBox_Coordinate = []
    for contour in contours:
       
       x,y,w,h = cv.boundingRect(contour)
       #print(f"BBox Coordinate x :{x}, y : {y}, w : {w}, h :{h}")
       BBox_Coordinate.append([x,y,w,h])
    
    BBox_Coordinate.sort(key=lambda bbox: bbox[0])
    
    if len(BBox_Coordinate) > 0:
        BBox_Coordinate.pop(0)
    
    
    main_bbox_x_max_range = -1
    count = -1
    cropped_images = []
    for idx, bbox in enumerate(BBox_Coordinate):
           
       x,y,w,h = bbox[0],bbox[1],bbox[2],bbox[3]            
       bbox_area = (w * h)
       main_next_bbox_x_min_range = x

       print("bbox_Area",bbox_area)
       print("h/w",h/w)
       print("H",h)
       print("y AND y+h",y,y+h)
       #print("interval",main_next_bbox_x_min_range - main_bbox_x_max_range)


       if (main_next_bbox_x_min_range >= main_bbox_x_max_range-5) and (bbox_area >= 9700.0 and bbox_area < (1350-60)*(700-157)) and (h/w <= 7.5):   # 제대로 잘리는 부분
       #if (main_next_bbox_x_min_range >= main_bbox_x_max_range-5) and (bbox_area >= 11000.0 and bbox_area < (1350-60)*(700-157)) and (h/w <= 7.5):    # 제대로 안 잘리는 부분    
       
            print("-----------------------------")
            #print("h/w",h/w)
            #print("y AND y+h",y,y+h)
            #print("interval",main_next_bbox_x_min_range - main_bbox_x_max_range)

            #cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)  # 초록색으로 Bounding Box 표시, 두께는 2
            #cv.imshow(f"{label}",image)
            #cv.waitKey(0)     
        
            main_bbox_x_max_range = main_next_bbox_x_min_range + w    
            count += 1

            roi = image[y+1:y+h-1, x+1:x+w-1]
            cropped_images.append(roi)
            
            # Save the cropped image
            output_path = os.path.join(f"{output_folder}", f"{label}_{count}.png")
            cv.imwrite(output_path, roi)
            
    cv.destroyAllWindows()
    return count == 5

        

    
# 사용 예제
images,labels = load_images_from_folder(folder_path)

for image,label in zip(images,labels):
  
  result = split_6_imgs_and_save(image,label)

  if result == False:
      print(f"label : {label}이 정상 분할에 실패")

      
