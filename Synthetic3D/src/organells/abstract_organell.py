import abc
from copy import deepcopy

from Synthetic3D.src.hard.structure.vector import Vector, is_zero
from Synthetic3D.src.hard.structure.shells import FrameShell


class Organell(abc.ABC):
    """
    Общий класс для подвижных органелл. Переменные frame и shell не зависят от смещения и реализуются
    относительно собственного центра координат ((0, 0, 0) - центр органеллы), но повернутых в мировой системе на угол
    angle, а view_poligon содержит треугольники для рисования в смещенных координатах на вектор position.

    Arguments:
        comment     : str? - поле для комментария каждой конкретной органеллы для возможности
                    создания уникальных именованных объектов.
        frame       : Frame - класс для хранения центральных точек органеллы (центры везикул, центральная линия
                    митохондрии, центральная линия аксона, центр ПСП и т.д.)
        shell       : Shell - класс для хранения грубых данных оболочки (в первом приближении без сглаживания)
        view_poligon: list[triangle] - список треугольников для рисования в нужных координатах и ориентации

        angle       : (float, float, float) - текущий угол поворота органеллы, спроецированный по осям X, Y и Z.
        position    : (int, int, int) - текущая позиция органеллы в пространстве

        view_data   : ViewData - переменная для храниения информации для рисования (цвет, толщина линий и т.д)

    Methods: ################################################################################################################## нуждается в доработке обобщенных методов
        SetPosition(new_pos) - функция изменения положения в пространстве в положение new_pos (int, int, int).
                               *смещение положения должно вычисляеться как новое минус старое
        Rotate(angle_delta) - функция поворота органеллы на угол angle_delta, заданный как (float, float, float)

        Draw(data) - функция рисования органеллы в пространстве data

    """

    def __init__(self):
        self.comment = ""

        self.shell = FrameShell()

        self.view_shell = None

        self.angle = Vector()
        self.position = Vector()

        self.params = {"color": (255, 0, 0),  # красный
                       "color_inner": (0, 255, 0),
                       "mask_color": 255}

        self.num_partition_of_triangles = 0
        self.warnings = []
    # ABSTRACT
    @abc.abstractmethod
    def _check_and_set_default_params(self) -> list[str]: pass # задает параметры генерации класса, если они не были заданы

    @abc.abstractmethod
    def _Create(self) -> list[str]: pass # генерирует форму, возвращает список с варнингами

    @abc.abstractmethod
    def Draw(self, data): pass  # реализация функции отрисовки на входных данных

    @abc.abstractmethod
    def DrawMask(self, mask_data, color=None): pass  # реализация функции отрисовки маски на входных данных

    @abc.abstractmethod
    def DrawArea(self, cell_data, color) -> list[Vector]: pass  # реализация функции заполнения уникальной территории

    # UNION METHODS
    def _CalculateViewData(self): # создает треугольники в нужной ориентации для рисования    №№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№ возможно не нужен, но сейчас используется для создания view_shell
        self._Calculate_Turn()

    ############################################################################################## В текущей реализации 1 треугольник дробится в 4
    def Partition_of_triangles(self, number_of_iteration):
        if self.view_shell is None:
            self.view_shell = deepcopy(self.shell)
        self.view_shell.Partition_of_triangles(number_of_iteration)

    def _Calculate_Shift(self): # смещает координаты точек на self.position
        self.view_shell = deepcopy(self.shell)
        if not is_zero(self.position):
            self.view_shell.shift_coords_by_value(self.position)
        #self.Partition_of_triangles(self.num_partition_of_triangles)

    def ChangePosition(self, delta_position): # функция переноса в органеллы на вектор смещения
        self.position += delta_position
        if self.view_shell is None:
            self._Calculate_Shift()
        else:
            self.view_shell.shift_coords_by_value(delta_position)

    def SetPosition(self, new_pos): # функция переноса в органеллы в точку
        if self.view_shell is None:
            self.position = new_pos
            self._Calculate_Shift()
        else:
            delta_position = new_pos - self.position
            self.position = new_pos
            self.view_shell.shift_coords_by_value(delta_position)

    def _Calculate_Turn(self):  # поворачивает точки относительно (0,0,0) на угол self.alpha
        self.shell.rotate_coords_and_normals(self.angle)
        self._Calculate_Shift()

    def Rotate(self, delta_angle:Vector): # функция поворота на заданный угол
        if not is_zero(delta_angle):
            self.angle = delta_angle + self.angle
            self._Calculate_Turn()

    def SetAngle(self, new_angle:Vector): # функция поворота в нужный угол
        if new_angle != self.angle:
            self.Rotate(new_angle - self.angle)

    def SetPositionAndAngle(self, pos, angle):
        delta_angle = angle - self.angle
        if not is_zero(delta_angle):
            self.position = pos
            self.angle = delta_angle + self.angle
            self._Calculate_Turn()
        else:
            delta_pos = pos - self.position
            if not is_zero(delta_pos):
                self.ChangePosition(delta_pos)
