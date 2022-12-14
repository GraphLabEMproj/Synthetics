import numpy as np
import cv2

import math

if __name__ == "__main__":
    import sys
    sys.path.append('.')

from src.container.spline import *
from src.container.subclass import *
from settings import PARAM, uniform_int

class Mitohondrion:
    def __init__(self):
        self.type = "Mitohondrion"

        color = uniform_int(
            PARAM['mitohondrion_shell_color_mean'],
            PARAM['mitohondrion_shell_color_std'])
        self.color = (color, color, color) #/// подобрать цвета, сделать заливку текстурой
                

        addColor = uniform_int(
            PARAM['mitohondrion_back_color_mean'],
            PARAM['mitohondrion_back_color_std'])
        self.addColor = (addColor, addColor, addColor) # основной цвет текстуры
        
        self.nowPen = Pen(self.color, np.random.randint(3, 5+1))    
        self.nowBrush = None
        self.texture = None

        self.angle = 0
        self.main_len = 0
        self.numberPoints = 0     
        self.centerPoint = [0, 0]
        self.Points = []
        self.PointsWithOffset = []
        self.addPoints = []            # точки с смещением для имитации косоро среза
        self.addPointsWithOffset = []

        self.Create()

    def Create(self):
        
        min_len = 26
        max_len = 196
                
        main_len = np.random.randint(min_len, max_len)
        self.main_len = main_len
        
        min_r = np.random.randint(5, 27)
        max_r = np.random.randint(27, 80)
        
        self.numberPoints = 2 + 2 * np.random.randint(2, 4+1)
        
        tPoints = []
        # Добавление первой главной точки
        startMPoint = [-main_len//2, 0]
        tPoints.append(startMPoint.copy())
        
        half_len = (self.numberPoints - 2)//2
        step = main_len/ (half_len+1)
        
        # во избежании создания крайне извилистой или вытянутой митохондрии вводится ограничение на изменение крутизны её извилитости
        start_y_value = np.random.randint(min(max(step, min_r),max_r)//2 , min(max_r, main_len//2)+1)

        # заполнение первой половины митохондрии
        for i in range(half_len):
           addPointX = startMPoint[0] + step * (i+1)
           addPointY = start_y_value
           #print(addPointX, addPointY) 
           tPoints.append([addPointX, addPointY])
           
           start_y_value = min(max(start_y_value + np.random.randint(-max_r//2, max_r//2 + 1), min_r), max_r)
       
        # Добавление второй главной точки
        endMPoint = [main_len//2, 0]
        tPoints.append(endMPoint.copy())
        
        # заполнение второй половины митохондрии
        start_y_value = np.random.randint(min(max(step, min_r),max_r)//2 , min(max_r, main_len//2)+1)
        for j in range(half_len):
           addPointX = endMPoint[0] - step * (j+1)
           addPointY = -start_y_value
           #print(addPointX, addPointY)
           tPoints.append([addPointX, addPointY])
        
           start_y_value = min(max(start_y_value + np.random.randint(-max_r//2, max_r//2 + 1), min_r), max_r)
        
        # поворот митохондрии
        self.angle = np.random.randint(0, 90) 
        #angle = self.angle * (math.pi/180)
        
        #print(self.numberPoints, half_len, len(tPoints))
        self.Points = tPoints

        #добавление доп. точек
        if np.random.random() < 0.5:            
            size_dop = np.random.randint(3,4+1) * 2
            
            startMPointWithOffset = [startMPoint[0] - size_dop//2, 0]
            
            tAddPoints1 = [
                           tPoints[1],
                           startMPointWithOffset,
                           tPoints[-1],
                           
                           tPoints[1],
                           [startMPoint[0] + size_dop//2, startMPoint[1]],
                           tPoints[-1]
                           ]
                           
            endMPointWithOffset = [endMPoint[0] + size_dop//2, 0]
            
            tAddPoints2 = [
                           tPoints[half_len],
                           endMPointWithOffset,
                           tPoints[half_len+2],
                           
                           tPoints[half_len],
                           [endMPoint[0] + size_dop//2, endMPoint[1]],
                           tPoints[half_len+2]
                           ]
                           
            tAddPoints = tAddPoints1 + tAddPoints2
            
            self.addPoints = tAddPoints 

        self.setRandomAngle(0, 0)

    def ChangePositionPoints(self):
        self.PointsWithOffset = []
        self.addPointsWithOffset = []

        for point in self.Points:
            self.PointsWithOffset.append([self.centerPoint[0]+point[0], self.centerPoint[1]+point[1]])
            
        for point2 in self.addPoints:
            self.addPointsWithOffset.append([self.centerPoint[0]+point2[0], self.centerPoint[1]+point2[1]])

    def NewPosition(self, x, y):
        self.centerPoint[0] = x
        self.centerPoint[1] = y

        self.ChangePositionPoints()
       
    def setRandomAngle(self, min_angle = 0, max_angle = 90, is_singned_change = True):  
    
        if np.random.random() < 0.5 and is_singned_change:
            sign = -1
        else:
            sign = 1
            
        self.angle = (self.angle + np.random.randint(min_angle, max_angle+1) * sign) %360
        
        change_angle = self.angle * (math.pi/180)
        
        tPoints = []
        for point in self.Points:
            x = int(round(point[0] * math.cos(change_angle) - point[1] * math.sin(change_angle)))
            y = int(round(point[0] * math.sin(change_angle) + point[1] * math.cos(change_angle)))
            tPoints.append([x,y])

        tAddPoints = []
        for point2 in self.addPoints:
            x = int(round(point2[0] * math.cos(change_angle) - point2[1] * math.sin(change_angle)))
            y = int(round(point2[0] * math.sin(change_angle) + point2[1] * math.cos(change_angle)))
            tAddPoints.append([x,y]) 
       
        self.Points = tPoints
        self.addPoints = tAddPoints   
        self.ChangePositionPoints()
       
    def CreateTexture(self, image):

        self.texture = np.full((*image.shape[0:2],3), self.addColor, np.uint8)
        #self.texture[:,:] = (0,0,255)
        
        cristae_color = uniform_int(
            PARAM['mitohondrion_cristae_color_mean'],
            PARAM['mitohondrion_cristae_color_std'])
        cristae_color = (cristae_color,cristae_color,cristae_color)
        
        forecolor = uniform_int(
            PARAM['mitohondrion_cristae_shell_color_mean'],
            PARAM['mitohondrion_cristae_shell_color_std'])        
        forecolor = (forecolor,forecolor,forecolor)

        now_x = 10

        while now_x < image.shape[1] - 10:
            now_y = 10
            while now_y < image.shape[1] - 10:
                len_line = np.random.randint(0, self.main_len // 2)

                start_pos = [now_x + np.random.randint(-2,3), now_y]
                enf_pos = [now_x + np.random.randint(-2,3), now_y+len_line]
                
                self.texture = cv2.line(self.texture, start_pos, enf_pos, forecolor, 5)
                
                if len_line == 0: # генерация черной точки
                    if np.random.random() < 0.5: 
                        self.texture = cv2.line(self.texture, start_pos, enf_pos, (1,1,1), 5)
                        self.texture = cv2.line(self.texture, start_pos, enf_pos, (1,1,1), 2)
                    else:
                        self.texture = cv2.line(self.texture, start_pos, enf_pos, (cristae_color), 2)
                else:
                    self.texture = cv2.line(self.texture, start_pos, enf_pos, (cristae_color), 2)

                #step y
                step_y = np.random.randint(7,20)
                now_y = now_y + len_line + step_y
                
            #step x
            step_x = np.random.randint(12,18)
            now_x += step_x

        (h, w) = image.shape[:2]
        center = (int(w / 2), int(h / 2))
        rotation_matrix = cv2.getRotationMatrix2D(center, 90 - self.angle, 1.5)
        self.texture = cv2.warpAffine(self.texture, rotation_matrix, (w, h))
        
        #print(self.texture.shape)
        self.texture[self.texture[:,:,0] == 0] = self.addColor

        #cv2.imshow("texture", self.texture)
        #cv2.waitKey()
        self.nowBrush = Brush(brush = self.texture, typeFull = "texture")

    def Draw(self, image, layer_drawing = True):
        # Основная рисующая фукция
        if layer_drawing:
            self.CreateTexture(image)

        draw_image = image.copy()
                
                
        self.nowBrush.FullBrush(draw_image, self.PointsWithOffset)
                
        if len(self.addPointsWithOffset) != 0:
            fill_texture_2_poligons(draw_image, self.addPointsWithOffset[0:3], self.addPointsWithOffset[3:6], self.nowPen.color, self.nowPen.sizePen+1)
            fill_texture_2_poligons(draw_image, self.addPointsWithOffset[6:9], self.addPointsWithOffset[9:12], self.nowPen.color, self.nowPen.sizePen+1)
            
        spline_line(draw_image, self.PointsWithOffset, self.nowPen.color, self.nowPen.sizePen) 

        return draw_image
        
    def setDrawParam(self):
        self.nowPen.color = self.color 
        self.nowBrush = Brush(brush = self.texture, typeFull = "texture")
        
    def setMaskParam(self):
        self.nowPen.color = (255,255,255)
        self.nowBrush = Brush((255,255,255))
        
    def setMaskBoarderParam(self):
        self.nowPen.color = (255,255,255)
        self.nowBrush =  Brush((0,0,0))

    def DrawMask(self, image):
        #Смена цветов для рисования маски
        self.setMaskParam()
        mask = self.Draw(image, layer_drawing = False)
        self.setDrawParam()
        
        return mask

    def DrawMaskBoarder(self, image):
        #Смена цветов для рисования маски
        self.setMaskBoarderParam()
        mask = self.Draw(image, layer_drawing = False)
        self.setDrawParam()
        
        return mask           
            
    def DrawUniqueArea(self, image, small_mode = False):
        # Функция создающая маску с немного большим отступом для алгорима случайного размещения новых органнел без пересечения 
        ret_image = image.copy()
        ret_image = ret_image.astype(int)
        
        draw_image = np.zeros(image.shape, np.uint8) 
        draw_image = self.DrawMask(draw_image)
        
        if small_mode == False:
            kernel = np.array([[0, 1, 0],
                               [1, 1, 1],
                               [0, 1, 0]], dtype=np.uint8)
                               
            draw_image = cv2.dilate(draw_image,kernel,iterations = 4)     
                    
        ret_image = ret_image + draw_image
        ret_image[ret_image[:,:,:] > 255] = 255
        ret_image = ret_image.astype(np.uint8)
        
        return ret_image
        

def testMitohondrion():
    q = None

    print("Press button 'Q' or 'q' to exit")
    while q != ord('q') and q != ord("Q"):
        
        color = np.random.randint(182, 200)
        
        img = np.full((512,512,3), (color,color,color), np.uint8)
        
        for i in range(0, 512, 32):
            img[i:i+15,:,:] += 30
        
        mitohondrion = Mitohondrion()
        
        mitohondrion.NewPosition(256,256)
        
        print("centerPoint", mitohondrion.centerPoint)
        print("Points", mitohondrion.Points)
        print("PointsWithOffset", mitohondrion.PointsWithOffset)
        
        print("addPoints", mitohondrion.addPoints)
        print("addPointsWithOffset", mitohondrion.addPointsWithOffset)
        
        img1 = mitohondrion.Draw(img)

        mask = np.zeros((512,512,3), np.uint8)
        
        tecnicalMask = np.zeros((512,512,3), np.uint8)
        
        maskMito = mitohondrion.DrawMask(mask.copy())
        img2 = mitohondrion.Draw(img)
        tecnicalMask = mitohondrion.DrawUniqueArea(tecnicalMask)
        
        for point in mitohondrion.PointsWithOffset:
            cv2.circle(img1, point, 3, (0, 0, 255), 2)
            
        for point in mitohondrion.addPointsWithOffset:
            cv2.circle(img2, point, 3, (0, 255, 0), 2)
        
        Boarder = mitohondrion.DrawMaskBoarder(mask)
        
        cv2.imshow("img", img1)
        cv2.imshow("mask", maskMito)
        cv2.imshow("img2", img2)
        cv2.imshow("tecnicalMask", tecnicalMask)
        cv2.imshow("Boarder", Boarder)
        
        q = cv2.waitKey()

if __name__ == "__main__":
    testMitohondrion()


