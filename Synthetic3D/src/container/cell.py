import numpy as np
from Synthetic3D.src.hard.structure.vector import Vector
from Synthetic3D.src.hard.drawing_and_filliing.draw_element import edge_check_3D
from Synthetic3D.src.organells.empty_organelle import EmptyOrganelle

from Synthetic3D.src.utilities.check_of_params import check_param
from Synthetic3D.src.container.cell_operation import ExpansionOfPoint26Dir_v3, ExpansionOfPoint6Dir

class Cell():
    """
    Класс представляющий клетку и содержащий код для её расширения. Стартует из зоны вокруг органелл.
    index - метка данной клетки в датасете. Отрицательное значение обозначает границу.

    """

    def __init__(self, index, list_of_organels=[EmptyOrganelle()], params={}):
        self.warnings = []
        self.list_of_organells = list_of_organels
        self.work_points = []
        self.index = index
        self.params = params.get("Cell", {})

        self.warnings += check_param(self.params, "probability_of_expansion", 0.5)


    # рисует минимальную маску для разрастания регионов, полностью покрывая внутренние органеллы
    ####################################################################################################################### Нуждается в улучшении
    def DrawKernelOfArea(self, data):
        d,h,w = data.shape[:3]
        for organell in self.list_of_organells:
            t_work_points = organell.DrawArea(data, self.index)
            # отсеить краевые и перевести в целое, так как далее работа только с индексами
            for point in t_work_points:
                if not edge_check_3D(*point, w=w, h=h, d=d):
                    self.work_points.append(np.round(point).astype(int))

    ######################################################################### Подумать над необходимостью увеличения каналлов
    def ExpansionOfPoint26Dir(self, data, position: Vector | np.ndarray):
        assert len(data.shape) == 3 or data.shape[3] == 1, "Данные для алгоритма разрастания должны иметь 3 оси и 1 канал."
        d, h, w = data.shape
        p_x, p_y, p_z = position
        next_iteration_work_points = []
        repit_check_now_point = False

        for i_z in (-1, 0, 1):
            n_z = p_z + i_z
            for i_x in (-1, 0, 1):
                n_x = p_x + i_x
                for i_y in (-1, 0, 1):
                    n_y = p_y + i_y
                    if p_x == n_x and p_y == n_y and p_z == n_z: # текущая точка нам не интересна. только окресность
                        continue
                    elif not edge_check_3D(n_x,n_y,n_z, w, h, d): # если мы внутри data
                        # если не занято, то занимаем с некоторой вероятностью, иначе отправляем на пересмотр
                        if data[n_z, n_y, n_x] == 0:
                            # 1/3 для углов, 1/2 для середин ребер и 1 для середин граней
                            if np.random.random() < 1/(abs(i_z) + abs(i_y) + abs(i_x)):
                                next_iteration_work_points.append(Vector(n_x, n_y, n_z, dtype=int))
                                data[n_z, n_y, n_x] = self.index
                            else:
                                repit_check_now_point = True

                        # попали в соседа self.index или в свою границу
                        elif data[n_z, n_y, n_x] == self.index or data[n_z, n_y, n_x] == -self.index:
                            continue
                        # попали в гранчное состояние:
                        #   - попали в другой регион
                        #   - дотянулись до чужой границы
                        # говорим что текущая позиция - граница этой клетки если она уже не обозначена
                        else:
                            if data[p_z, p_y, p_x] != -self.index:
                                data[p_z, p_y, p_x] = -self.index
                            else:
                                continue

            if repit_check_now_point:
                next_iteration_work_points.append(Vector(p_x, p_y, p_z, dtype=int))

        return next_iteration_work_points

    def ExpansionOfRegion(self, data):
        assert len(data.shape) == 3 or data.shape[3] == 1, "Данные для алгоритма разрастания должны иметь 3 оси и 1 канал."
        index_kernel = np.array([
            (dz, dy, dx)
            for dz in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dx in (-1, 0, 1)
            if not (dz == 0 and dy == 0 and dx == 0)
        ], dtype=int)
        probability_kernel = np.array([
            1 / (abs(dz) + abs(dy) + abs(dx))
            for dz in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dx in (-1, 0, 1)
            if not (dz == 0 and dy == 0 and dx == 0)
        ])

        if len(self.work_points) > 0:
            self.work_points = np.array(self.work_points, dtype=np.int32)
            new_points = []
            for work_point in self.work_points:
                new_points += ExpansionOfPoint26Dir_v3(data, work_point, self.index, index_kernel, probability_kernel)
                #new_points += ExpansionOfPoint6Dir(data, work_point, self.index, self.params["probability_of_expansion"])

            self.work_points = new_points
            return True
        else:
            return False

    def SetPositionAndAngle(self, pos, angle):
        for org in self.list_of_organells:
            org.SetPositionAndAngle(pos, angle)
