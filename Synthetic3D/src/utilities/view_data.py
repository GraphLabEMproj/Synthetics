import numpy as np
import vtk
from vtk.util import numpy_support

from Synthetic3D.src.hard.structure.vertex import Vertex

EdgeVisibility = False
intensity = 2.0


def get_voxel_actor_from_data(data):
    # Предположим, что data — ваш numpy массив
    assert 2 < len(data.shape)

    if len(data.shape) == 3 or data.shape[3] == 1:
        X, Y, Z = data.shape
        C = 1
    else:
        X, Y, Z, C = data.shape

    # Создаем сетку координат
    x = np.arange(X)
    y = np.arange(Y)
    z = np.arange(Z)
    X_grid, Y_grid, Z_grid = np.meshgrid(x, y, z, indexing='ij')

    # Векторизуем и объединяем координаты
    coords = np.column_stack((Z_grid.ravel(), Y_grid.ravel(), X_grid.ravel()))
    values = data.ravel() if C == 1 else data.reshape(-1, C)  # для RGB

    # Фильтруем точки, у которых значение != 0
    mask = values != 0 if C == 1 else np.any(values != 0, axis=1)
    filtered_coords = coords[mask]
    filtered_values = values[mask]

    # Создаем vtkPoints для только выбранных точек
    points_vtk = vtk.vtkPoints()
    points_vtk.SetNumberOfPoints(filtered_coords.shape[0])
    points_vtk.SetData(numpy_support.numpy_to_vtk(filtered_coords, deep=True))

    if C == 1:
    # Создаем vtkScalars для выбранных точек
        colors_vtk = numpy_support.numpy_to_vtk(filtered_values, deep=True, array_type=vtk.VTK_FLOAT)
        colors_vtk.SetName("scalars")
    else:
        # Этот блок нужно добавить, чтобы преобразовать массив RGB в vtk-совместимый формат:
        colors_vtk = vtk.vtkUnsignedCharArray()
        colors_vtk.SetNumberOfComponents(3)
        colors_vtk.SetNumberOfTuples(filtered_coords.shape[0])
        for i, color in enumerate(filtered_values):
            assert color.shape == (3,)
            colors_vtk.SetTuple3(i, int(color[0]), int(color[1]), int(color[2]))

    # Создаем polyData и добавляем точки и скаляры
    polyData = vtk.vtkPolyData()
    polyData.SetPoints(points_vtk)
    polyData.GetPointData().SetScalars(colors_vtk)

    # Создаем источник куба для глифа
    cubeSource = vtk.vtkCubeSource()
    cubeSource.SetXLength(1)
    cubeSource.SetYLength(1)
    cubeSource.SetZLength(1)

    # Настраиваем маппер
    glyphMapper = vtk.vtkGlyph3DMapper()
    glyphMapper.SetInputData(polyData)
    glyphMapper.SetSourceConnection(cubeSource.GetOutputPort())

    glyphMapper.SetScalarRange(0, 255)

    if C == 1:
        # Создаем таблицу цветовой карты для оттенков серого
        lookupTable = vtk.vtkLookupTable()
        lookupTable.SetNumberOfTableValues(256)
        lookupTable.SetRange(np.min(filtered_values), np.max(filtered_values))
        for i in range(256):
            gray = i / 255.0  # градации от 0 до 1
            lookupTable.SetTableValue(i, gray, gray, gray, 1.0)
        lookupTable.Build()
        glyphMapper.SetScalarModeToUsePointData()
        glyphMapper.SetColorModeToMapScalars()
        glyphMapper.SetLookupTable(lookupTable)
    else:
        glyphMapper.SetColorModeToDirectScalars()

    # Создаем актер для глифов
    glyphActor = vtk.vtkActor()
    glyphActor.SetMapper(glyphMapper)
    glyphActor.GetProperty().EdgeVisibilityOn()

    return glyphActor

def create_vector_actor(vertices, scale_factor=5.0):
    assembly = vtk.vtkAssembly()

    for vertex in vertices:
        start_point = vertex.position
        vector = vertex.normal

        length = np.linalg.norm(vector)
        if length == 0:
            continue

        arrow_source = vtk.vtkArrowSource()

        transform = vtk.vtkTransform()
        transform.Translate(start_point)

        default_direction = np.array([1, 0, 0])
        direction = np.array(vector) / length

        # Вычисляем ось вращения
        rotation_axis = np.cross(default_direction, direction)
        axis_norm = np.linalg.norm(rotation_axis)

        # Вычисляем угол в радианах
        angle_rad = np.arccos(np.clip(np.dot(default_direction, direction), -1.0, 1.0))
        angle_deg = np.degrees(angle_rad)

        if axis_norm > 1e-6:
            # Вращаем стрелку так, чтобы она указывала в направлении вектора
            transform.RotateWXYZ(angle_deg, rotation_axis[0], rotation_axis[1], rotation_axis[2])
        else:
            # Если векторы противоположны, повернуть на 180° вокруг произвольной перпендикулярной оси
            if np.dot(default_direction, direction) < 0:
                # Выбираем произвольную ось, перпендикулярную default_direction
                perp_axis = np.array([0, 1, 0])
                # Проверка, чтобы perp_axis не была параллельна default_direction
                if np.allclose(np.cross(default_direction, perp_axis), 0):
                    perp_axis = np.array([0, 0, 1])
                transform.RotateWXYZ(180, perp_axis[0], perp_axis[1], perp_axis[2])
            else:
                # Вектора совпадают, ничего не делаем
                pass

        mod_length = length * scale_factor
        transform.Scale(mod_length, mod_length, mod_length)

        transform_filter = vtk.vtkTransformPolyDataFilter()
        transform_filter.SetTransform(transform)
        transform_filter.SetInputConnection(arrow_source.GetOutputPort())
        transform_filter.Update()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(transform_filter.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0, 1, 0)

        assembly.AddPart(actor)
    return assembly

def get_actor_point(center):
    # Создаем источник сферы
    sphere_source = vtk.vtkSphereSource()

    # Задаем центр сферы (например, в координатах (x, y, z))
    sphere_source.SetCenter(*center)

    # Задаем радиус сферы
    radius = 10  # Задайте нужный радиус
    sphere_source.SetRadius(radius)

    # Создаем маппер и актор для отображения
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(sphere_source.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    return actor

def view_vtk_3D_data(data, vertex_list: list[Vertex]=None, center_point=None):
    glyphActor = get_voxel_actor_from_data(data)

    # Создаем рендерер и добавляем актер
    renderer = vtk.vtkRenderer()
    renderer.AddActor(glyphActor)
    renderer.SetBackground(0.1, 0.2, 0.4)

    if vertex_list is not None:
        vertices_actor = create_vector_actor(vertex_list)
        renderer.AddActor(vertices_actor)

    if center_point is not None:
        center_actor = get_actor_point(center_point)
        renderer.AddActor(center_actor)

    # Создаем окно рендера
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.AddRenderer(renderer)
    renderWindow.SetSize(800, 600)

    # Создаем интерактор
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(renderWindow)

    # Создаем интерактор
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(renderWindow)

    def EdgeVisibilityChange(obj, event):
        global EdgeVisibility
        if EdgeVisibility:
            glyphActor.GetProperty().EdgeVisibilityOff()
            EdgeVisibility = False
        else:
            glyphActor.GetProperty().EdgeVisibilityOn()
            EdgeVisibility = True
        renderWindow.Render()

    def on_default_style(obj, event):
        style = vtk.vtkInteractorStyleTrackballCamera()
        interactor.SetInteractorStyle(style)

    # Создаем источник света
    light = vtk.vtkLight()
    light.SetLightTypeToCameraLight()  # Связь с камерой
    global intensity
    light.SetIntensity(intensity)  # Увеличиваем яркость
    renderer.AddLight(light)

    # Функция для изменения яркости
    def set_light_brightness(value):
        global intensity # — значение от 0 до 1
        if value == "up":
            intensity += 1
        else:
            intensity -= 1
        light.SetIntensity(intensity)
        renderWindow.Render()
    def keypress_callback(obj, event):
        index = 0

        key = obj.GetKeySym()
        if key == 'a':
            EdgeVisibilityChange(obj, event)
        elif key == 's':
            on_default_style(obj, event)

        # Например, изменить яркость при нажатии клавиши "+" или "-"
        elif key == 'plus' or key == 'KP_Add':
            print("plus")
            set_light_brightness("up")  # максимально ярко
        elif key == 'minus' or key == 'KP_Subtract':
            print("minus")
            set_light_brightness("down")  # тускло

    interactor.AddObserver("KeyPressEvent", keypress_callback)


    # Запуск рендера
    renderWindow.Render()
    interactor.Initialize()
    interactor.Start()

# Пока что дорог как первый код, в релизной версии выпилить !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def view_vtk_3D_data_old(data):
    # Convert to VTK ImageData
    vtk_image_data = vtk.vtkImageData()
    vtk_image_data.SetDimensions(data.shape[0], data.shape[1], data.shape[2])
    vtk_image_data.GetPointData().SetScalars(vtk.util.numpy_support.numpy_to_vtk(data.ravel(), deep=True))

    threshold_points = vtk.vtkThresholdPoints()
    threshold_points.SetInputData(vtk_image_data)
    threshold_points.SetUpperThreshold(1)
    #threshold_points.SetLowerThreshold(0)
    threshold_points.Update()

    '''
    surface = vtk.vtkMarchingCubes()
    surface.SetInputConnection(poly_data.GetOutputPort())
    surface.ComputeNormalsOn()
    surface.SetValue(0, 1)
    '''

    poly_data = vtk.vtkPolyData()
    poly_data.SetPoints(threshold_points.GetOutput().GetPoints())

    cubeSorce = vtk.vtkCubeSource()
    cubeSorce.SetZLength(1)
    cubeSorce.SetXLength(1)
    cubeSorce.SetYLength(1)

    glyph3D = vtk.vtkGlyph3D()
    glyph3D.SetSourceConnection(cubeSorce.GetOutputPort())
    glyph3D.SetColorModeToColorByScalar()
    glyph3D.SetInputData(poly_data)
    glyph3D.SetVectorModeToUseNormal()
    glyph3D.Update()

    # Создание мэппера
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(glyph3D.GetOutputPort())
    mapper.SetColorModeToDirectScalars()
    mapper.SetScalarModeToUseCellData()
    # mapper.SetScalarModeToUsePointData()
    # mapper.SetScalarRange(0.0, 0.5)
    # mapper.ScalarVisibilityOn()
    # mapper.SetScalarModeToUsePointData()
    #
    mapper.Update()
    # mapper.

    # Создание актера
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    # actor.GetProperty().SetPointSize(5)
    actor.GetProperty().EdgeVisibilityOn()

    # Создание рендерера и окна для отображения
    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    # renderer.AddActor(axix_actor)
    # renderer.AddActor(actor_grid)
    renderer.SetBackground(0.1, 0.2, 0.4)

    window = vtk.vtkRenderWindow()
    window.AddRenderer(renderer)
    window.SetSize(800, 600)

    # Создание интерактивного отображения
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(window)
    interactor.Initialize()
    interactor.Start()
