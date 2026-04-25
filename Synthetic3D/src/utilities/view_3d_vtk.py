import os

import cv2
import numpy as np
import vtk
from vtk.util import numpy_support

def inner_filter3channel(data):
    data_filter = data.copy()

    for z in range(1, data.shape[0]-1):
        for y in range(1, data.shape[1] - 1):
            for x in range(1, data.shape[2] - 1):
                if any(data[z  , y, x] > 0) and \
                   any(data[z+1, y, x] > 0) and \
                   any(data[z-1, y, x] > 0) and\
                   any(data[z, y+1, x] > 0) and\
                   any(data[z, y-1, x] > 0) and\
                   any(data[z, y  , x+1] > 0) and\
                   any(data[z, y  , x-1] > 0):
                    data_filter[z, y, x] = 0
    print("data filtred !")
    return data_filter

def inner_filter(data):
    data_filter = data.copy()

    for z in range(1, data.shape[0]-1):
        for y in range(1, data.shape[1] - 1):
            for x in range(1, data.shape[2] - 1):
                if  data[z  , y, x] > 0 and\
                    data[z+1, y, x] > 0 and\
                    data[z-1, y, x] > 0 and\
                    data[z, y+1, x] > 0 and\
                    data[z, y-1, x] > 0 and\
                    data[z, y  , x+1] > 0 and\
                    data[z, y  , x-1] > 0:
                    data_filter[z, y, x] = 0
    print("data filtred !")
    return data_filter
def view_vtk_3D_data_mod_data(data):
    # Convert to VTK ImageData
    vtk_image_data = vtk.vtkImageData()
    vtk_image_data.SetDimensions(data.shape[2], data.shape[1], data.shape[0])
    vtk_image_data.GetPointData().SetScalars(vtk.util.numpy_support.numpy_to_vtk(data.ravel(), deep=True))

    print("Input image data dimensions:", vtk_image_data.GetDimensions())

    threshold_data = vtk.vtkImageThreshold()
    threshold_data.SetInputData(vtk_image_data)
    threshold_data.ThresholdByUpper(127)
    #threshold_data.ThresholdByLower(0)
    threshold_data.SetInValue(255)  # Значение для данных внутри диапазона
    threshold_data.SetOutValue(0)  # Значение для данных вне диапазона
    threshold_data.SetOutputScalarTypeToUnsignedChar()
    threshold_data.Update()

    threshold_points = vtk.vtkThresholdPoints()
    threshold_points.SetInputData(vtk_image_data)
    threshold_points.SetUpperThreshold(1)
    #threshold_points.SetLowerThreshold(0)
    threshold_points.Update()

    #geometry_filter = vtk.vtkImageDataGeometryFilter()
    #geometry_filter.SetInputConnection(threshold_points.GetOutputPort())
    #geometry_filter.Update()

    poly_data = vtk.vtkPolyData()
    poly_data.SetPoints(threshold_points.GetOutput().GetPoints())

    cubeSorce = vtk.vtkCubeSource()
    cubeSorce.SetZLength(1)
    cubeSorce.SetXLength(1)
    cubeSorce.SetYLength(1)

    glyph3D = vtk.vtkGlyph3D()
    glyph3D.SetSourceConnection(cubeSorce.GetOutputPort())
    glyph3D.SetInputData(poly_data)
    glyph3D.SetColorModeToColorByScalar()
    glyph3D.ScalingOff()

    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(5)
    lut.Build()
    for i in range(5):
        lut.SetTableValue(i, i / 5.0, 0, 1 - i / 5.0, 1)  # градиент


    marching = vtk.vtkFlyingEdges3D()
    marching.SetInputData(threshold_data.GetOutput())
    #surface.ComputeNormalsOn()
    marching.SetValue(0, 128)
    marching.Update()

    # Создание мэппера
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(glyph3D.GetOutputPort())
    #mapper.SetColorModeToDirectScalars()
    #mapper.SetScalarModeToUseCellData()
    # mapper.SetScalarModeToUsePointData()
    mapper.SetScalarRange(1, 5)  # диапазон значений для цвета
    mapper.ScalarVisibilityOn()
    mapper.SetLookupTable(lut)


    # Создание актера
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    # actor.GetProperty().SetPointSize(5)
    #actor.GetProperty().EdgeVisibilityOn()

    # Создание рендерера и окна для отображения
    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    # renderer.AddActor(axix_actor)
    # renderer.AddActor(actor_grid)
    renderer.SetBackground(0.1, 0.2, 0.4)

    window = vtk.vtkRenderWindow()
    window.AddRenderer(renderer)
    window.SetSize(2000, 1000)

    # Создание интерактивного отображения
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(window)
    interactor.Initialize()
    interactor.Start()


if __name__ == "__main__":
    path_to_data = "D:/Data/Unet_multiclass/data/Luchi_pp_train" + "/mitochondria"

    img_names = [name for name in os.listdir(path_to_data)]
    print(img_names)

    data_epfl = np.zeros((165, 768, 1024), dtype=np.uint8)

    for i, name in enumerate(img_names):
        img = cv2.imread(os.path.join(path_to_data, name), 0)
        print(img.shape, img.max(), img.min())

        data_epfl[i] = img

        #cv2.imshow("img", data_epfl[i])
        #cv2.waitKey()

    view_vtk_3D_data_mod_data(inner_filter(data_epfl))
