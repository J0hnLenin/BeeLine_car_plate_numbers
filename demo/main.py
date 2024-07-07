import cv2
import numpy as np
import os

INPUT_FOLDER_PATH = "./input_images"
OUTPUT_FOLDER_PATH = "./output_images"

""" Функция, выделяющая прямые на изображении""" 
def get_lines(contours):
  # Применяем преобразование Хафа
  lines = cv2.HoughLines(contours,1,np.pi/180,70) 
  if lines is None:
    # Если ничего не нашли, то снижаем порог
    lines = cv2.HoughLines(contours,1,np.pi/180,40)
  if lines is None:
    lines = cv2.HoughLines(contours,1,np.pi/180,20)
  if lines is None:
    lines = []
  return lines 

""" Функция, для преобразования прямых""" 
def reform_lines(img, lines):
  new_lines = []

  for i in range(len(lines)):
    line = []
    for rho,theta in lines[i]:
      q = 512
      a = np.cos(theta)
      b = np.sin(theta)
      x0 = a*rho
      y0 = b*rho
      x1 = int(x0 + q*(-b))
      y1 = int(y0 + q*(a))
      x2 = int(x0 - q*(-b))
      y2 = int(y0 - q*(a))

      if x1 == x2:
        x1 = int(x0 + 2*q*(-b))
        y1 = int(y0 + 2*q*(a))

      if x1!=x2:
        k = (y1-y2)/(x1-x2)
      else:
        k=0
      if np.isnan(k):
        k=0
      
      if abs(k)>0.45:
        # Избавляемся от вертикальных прямых
        continue
      
      b = y1-k*x1

      x1 = 0
      x2 = img.shape[1]

      y1 = k*x1+b
      y2 = k*x2+b

      line = int(x1), int(y1), int(x2), int(y2)
      new_lines.append(line)
  return new_lines
  
""" Функция для отбраывания лишних прямых"""
def filter_lines(lines):
  if len(lines) >= 2:
    top = 0
    bot = 0
    maxx = -1, -1
    minn = 100000000, 100000000 
    # Ищем свмую верхнюю и нижнюю прямые
    for i in range(len(lines)):
      x1, y1, x2, y2 = lines[i]
      #print(x1, y1, x2, y2)
      if (y1 > maxx[0] and y2 > maxx[1]) or (y1 > maxx[1] and y2 > maxx[0]) :
        maxx = y1, y2
        top = i
      elif (y1 < minn[0] and y2 < minn[1]) or (y1 < minn[1] and y2 < minn[0]) :
        minn = y1, y2
        bot = i
          
    best_lines = [lines[bot], lines[top]]
  else:
    best_lines = lines
  return best_lines

""" Поворот изображения на основании одной из прямых"""
def rotate_image(img, lines):
  if not lines:
    return img
  
  x1, y1, x2, y2 = lines[0]

  if x1!=x2:
    k = (y1-y2)/(x1-x2)
  
    if abs(x2 - x1) != 0:
      tg_alptha = abs(y2 - y1)/abs(x2 - x1)
      angle = np.sign(k)*np.rad2deg(np.arctan(tg_alptha))
    else:
      angle = 0

    h, w = img.shape[:2]
    center = (int(w / 2), int(h / 2))
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
    
    img = cv2.warpAffine(img, rotation_matrix, (w, h))
  return img

""" Функция для отрисовки прямых, нужна для отладки решения"""
def draw_lines(img, red_lines=[], green_lines=[]):
  for i in range(len(red_lines)):
      x1, y1, x2, y2 = red_lines[i]
      cv2.line(img,(x1,y1),(x2,y2),(255,0,0),2)

  for i in range(len(green_lines)):
    x1, y1, x2, y2 = green_lines[i]      
    cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
  return img

""" Функция для обрезки изображения"""
def crop_image(img):
  k = 50
  img = cv2.resize(img, (512+k,112+k))
  img = img[k//2:-k//2, k//2:-k//2, :]
  return img

def img_deformation(img):
  """ БЛОК АЛГОРИТМА ПО ДЕФОРМАЦИИ ИЗОБРАЖЕНИЯ """
  blur = cv2.GaussianBlur(img, (3, 3), 0) 
  img_grey = cv2.cvtColor(blur,cv2.COLOR_BGR2GRAY)

  img_contours = cv2.Canny(img_grey,140,200,apertureSize = 3)
  lines = get_lines(img_contours)
  
  lines = reform_lines(img, lines)
  
  best_lines = filter_lines(lines)
  #img = draw_lines(img, red_lines=best_lines)
  img = rotate_image(img, best_lines)
    
  img = crop_image(img)
  
  """               КОНЕЦ БЛОКА                """
  return img

def main():
    file_list = os.listdir(INPUT_FOLDER_PATH)
    image_files = [f for f in file_list if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    
    for i in range(len(image_files)):
        img_path = os.path.join(INPUT_FOLDER_PATH, image_files[i])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img_deformation(img)
        img = cv2.resize(img, (512,112))
    
        file_destination = os.path.join(OUTPUT_FOLDER_PATH, image_files[i])
        cv2.imwrite(file_destination, img)

if __name__ == '__main__':
    main()
