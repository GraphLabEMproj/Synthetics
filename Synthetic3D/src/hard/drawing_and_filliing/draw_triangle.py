from Synthetic3D.src.hard.drawing_and_filliing.draw_element import setVoxel3D
from Synthetic3D.src.hard.structure.vector import Vector, vector_to_int
from Synthetic3D.src.hard.drawing_and_filliing.draw_line_3d import draw_line_3D, generate_points_by_line_3D

def recurse_draw_voxel_triangle(data, gen_line_1, point2, point3, color, thickness):
    point1 = next(gen_line_1)

    # проверка на вырожденный случай
    p1p2 = (point2[0] - point1[0], point2[1] - point1[1], point2[2] - point1[2])
    p1p3 = (point3[0] - point1[0], point3[1] - point1[1], point3[2] - point1[2])

    # если точка 1 совпадает с точкой 2
    if (p1p2[0] == 0 and p1p2[1] == 0 and p1p2[2] == 0):
        # если точка 1 совпадает с точкой 2 и точкой 3
        if (p1p3[0] == 0 and p1p3[1] == 0 and p1p3[2] == 0):
            return setVoxel3D(data, point1[0], point1[1], point1[2], color, thickness)
        # если точка 1 совпадает с точкой 2 но не с точкой 3
        else:
            return draw_line_3D(data, point1, point3, color, thickness)
    # если точка 1 совпадает с точкой 3
    elif (p1p3[0] == 0 and p1p3[1] == 0 and p1p3[2] == 0):
        return draw_line_3D(data, point2, point3, color, thickness)

    # если вектора коллениарны (их векторное = 0), то в зависимости от направления одна точка отбрасывается
    elif p1p2[2] * p1p3[1] - p1p2[1] * p1p3[2] == 0 and \
            p1p2[0] * p1p3[2] - p1p2[2] * p1p3[0] == 0 and \
            p1p2[1] * p1p3[0] - p1p2[0] * p1p3[1] == 0:
        if p1p2[0] * p1p3[0] + p1p2[1] * p1p3[1] + p1p2[2] * p1p3[2] < 0:
            return draw_line_3D(data, point2, point3, color, thickness)
        elif p1p2[0] < p1p3[0] or p1p2[1] < p1p3[1] or p1p2[2] < p1p3[2]:
            return draw_line_3D(data, point1, point3, color, thickness)
        else:
            return draw_line_3D(data, point1, point2, color, thickness)

    # основной алгоритм
    else:
        gen_line_2 = generate_points_by_line_3D(point2, point3)

        work_gen_1 = True
        work_gen_2 = True

        # использование взятой изначально точки для проверок из генератора 1 (своеобразный do while)
        point2_temp = next(gen_line_2)
        draw_line_3D(data, point2_temp, point1, color, thickness)
        # выходим если одна из сторон закончилась
        if point1 == point3:
            work_gen_1 = False
        if point2_temp == point3:
            work_gen_2 = False

        # Поэтапное заполнение воксельными линиями пока не закончится одна из них
        while work_gen_1 and work_gen_2:
            # на всякий случай для отслеживания ошибочных ситуаций
            try:
                iter_point_line1 = next(gen_line_1)
                iter_point_line2 = next(gen_line_2)
            except StopIteration as e:
                print("Warning !!! Нет точек для полигона в R fun !")
                print(f"{work_gen_1}, {work_gen_2}")
                print(f"{iter_point_line1}, {iter_point_line2}, points {point1}, {point2}, {point3}, {point2_temp}")
                break

            draw_line_3D(data, iter_point_line2, iter_point_line1, color, thickness)

            # выходим если одна из сторон закончилась
            if iter_point_line1 == point3:
                work_gen_1 = False
            if iter_point_line2 == point3:
                work_gen_2 = False

        # Если что-то осталось - заполнить точками из оставшейся линии
        # Если закончилась первая линия (Алгоритм в точке P2), но вторая ещё имеет точки на P1P3
        if work_gen_2:
            while iter_point_line2 != point3:
                setVoxel3D(data, iter_point_line2[0], iter_point_line2[1], iter_point_line2[2], color, thickness)
                # на всякий случай для отслеживания ошибочных ситуаций
                try:
                    iter_point_line2 = next(gen_line_2)
                except StopIteration as e:
                    print("Warning !!! Нет точек для оставшейся стороны 3-1 !")
                    break

        # Если закончилась вторая линия (Алгоритм в точке P2), но первая ещё имеет точки на P2P3
        elif work_gen_1:
            while iter_point_line1 != point3:
                setVoxel3D(data, iter_point_line1[0], iter_point_line1[1], iter_point_line1[2], color, thickness)
                # на всякий случай для отслеживания ошибочных ситуаций
                try:
                    iter_point_line1 = next(gen_line_1)
                except StopIteration as e:
                    print("Warning !!! Нет точек для оставшейся стороны 2-1 !")
                    break

        return data

def draw_voxel_triangle(data, point1:Vector, point2:Vector, point3:Vector, color, thickness=0):
    point1 = vector_to_int(point1)
    point2 = vector_to_int(point2)
    point3 = vector_to_int(point3)

    # проверка на вырожденный случай
    p1p2 = (point2[0] - point1[0], point2[1] - point1[1], point2[2] - point1[2])
    p1p3 = (point3[0] - point1[0], point3[1] - point1[1], point3[2] - point1[2])

    # если точка 1 совпадает с точкой 2
    if (p1p2[0] == 0 and p1p2[1] == 0 and p1p2[2] == 0):
        # если точка 1 совпадает с точкой 2 и точкой 3
        if (p1p3[0] == 0 and p1p3[1] == 0 and p1p3[2] == 0):
            return setVoxel3D(data, point1[0], point1[1], point1[2], color, thickness)
        # если точка 1 совпадает с точкой 2 но не с точкой 3
        else:
            return draw_line_3D(data, point1, point3, color, thickness)
    # если точка 1 совпадает с точкой 3
    elif (p1p3[0] == 0 and p1p3[1] == 0 and p1p3[2] == 0):
        return draw_line_3D(data, point2, point3, color, thickness)

    # если вектора коллениарны (их векторное = 0), то в зависимости от направления одна точка отбрасывается
    elif p1p2[2] * p1p3[1] - p1p2[1] * p1p3[2] == 0 and \
         p1p2[0] * p1p3[2] - p1p2[2] * p1p3[0] == 0 and \
         p1p2[1] * p1p3[0] - p1p2[0] * p1p3[1] == 0:
        if p1p2[0] * p1p3[0] + p1p2[1] * p1p3[1] + p1p2[2] * p1p3[2] < 0:
            return draw_line_3D(data, point2, point3, color, thickness)
        elif p1p2[0] < p1p3[0] or p1p2[1] < p1p3[1] or p1p2[2] < p1p3[2]:
            return draw_line_3D(data, point1, point3, color, thickness)
        else:
            return draw_line_3D(data, point1, point2, color, thickness)

    # основной алгоритм
    else:
        # V1
        gen_line_1 = generate_points_by_line_3D(point1, point2)
        gen_line_2 = generate_points_by_line_3D(point1, point3)

        work_gen_1 = True
        work_gen_2 = True

        setVoxel3D(data, point1[0], point1[1], point1[2], color, thickness)

        # Поэтапное заполнение воксельными линиями пока не закончится одна из них
        while work_gen_1 and work_gen_2:
            # на всякий случай для отслеживания ошибочных ситуаций
            try:
                iter_point_line1 = next(gen_line_1)
                iter_point_line2 = next(gen_line_2)
            except StopIteration as e:
                print("Warning !!! Нет точек для полигона !")
                break

            draw_line_3D(data, iter_point_line1, iter_point_line2, color, thickness)

            # выходим если одна из сторон закончилась
            if iter_point_line1 == point2:
                work_gen_1 = False
            if iter_point_line2 == point3:
                work_gen_2 = False

        # Если что-то осталось - заполнить рекурсивно
        # Если закончилась первая линия (Алгоритм в точке P2), но вторая ещё имеет точки на P1P3
        if work_gen_2:
            del gen_line_1 # для экономии памяти
            return recurse_draw_voxel_triangle(data, gen_line_2, point2, point3, color, thickness)
        # Если закончилась вторая линия (Алгоритм в точке P2), но первая ещё имеет точки на P2P3
        elif work_gen_1:
            del gen_line_2 # для экономии памяти
            return recurse_draw_voxel_triangle(data, gen_line_1, point3, point2, color, thickness)
        # дошли до конца
        else:
            return data

# должен быть немного быстрее, но оставляет дыры в 1 пиксель.
def draw_voxel_triangle_v2(data, point1, point2, point3, color, thickness):
    print("WARNING ! При использовании функции draw_voxel_triangle_v2 могут быть пропуски (дырки) в 1 воксель !")
    # проверка на вырожденный случай
    p1p2 = (point2[0] - point1[0], point2[1] - point1[1], point2[2] - point1[2])
    p1p3 = (point3[0] - point1[0], point3[1] - point1[1], point3[2] - point1[2])

    # если точка 1 совпадает с точкой 2
    if (p1p2[0] == 0 and p1p2[1] == 0 and p1p2[2] == 0):
        # если точка 1 совпадает с точкой 2 и точкой 3
        if (p1p3[0] == 0 and p1p3[1] == 0 and p1p3[2] == 0):
            return setVoxel3D(data, point1[0], point1[1], point1[2], color, thickness)
        # если точка 1 совпадает с точкой 2 но не с точкой 3
        else:
            return draw_line_3D(data, point1, point3, color, thickness)
    # если точка 1 совпадает с точкой 3
    elif (p1p3[0] == 0 and p1p3[1] == 0 and p1p3[2] == 0):
        return draw_line_3D(data, point2, point3, color, thickness)

    # если вектора коллениарны (их векторное = 0), то в зависимости от направления одна точка отбрасывается
    elif p1p2[2] * p1p3[1] - p1p2[1] * p1p3[2] == 0 and \
            p1p2[0] * p1p3[2] - p1p2[2] * p1p3[0] == 0 and \
            p1p2[1] * p1p3[0] - p1p2[0] * p1p3[1] == 0:
        if p1p2[0] * p1p3[0] + p1p2[1] * p1p3[1] + p1p2[2] * p1p3[2] < 0:
            return draw_line_3D(data, point2, point3, color, thickness)
        elif p1p2[0] < p1p3[0] or p1p2[1] < p1p3[1] or p1p2[2] < p1p3[2]:
            return draw_line_3D(data, point1, point3, color, thickness)
        else:
            return draw_line_3D(data, point1, point2, color, thickness)

    # основной алгоритм
    else:
        # v2 ############### есть пропуски в линиях заливки
        gen_line_1 = generate_points_by_line_3D(point2, point1)
        gen_line_2 = generate_points_by_line_3D(point3, point1)

        work_gen_1 = True
        work_gen_2 = True

        draw_line_3D(data, point2, point3, color, thickness)

        # Поэтапное заполнение воксельными линиями пока не закончится одна из них
        while True:
            # на всякий случай для отслеживания ошибочных ситуаций
            try:
                iter_point_line1 = next(gen_line_1)
                iter_point_line2 = next(gen_line_2)
            except StopIteration as e:
                print(
                    f"Warning !!! Нет точек для полигона {point1}, {point2}, {point3}! {iter_point_line1}, {iter_point_line2}")
                break

            # print(f"Point 1 {point1}, Point 2 {point2}, Point 3 {point3}")
            # print(f"Point 2-1 {iter_point_line1}")
            # print(f"Point 3-1 {iter_point_line2}")

            # выходим если одна из сторон закончилась
            if iter_point_line1 == point1:
                work_gen_1 = False
            if iter_point_line2 == point1:
                work_gen_2 = False

            if not (work_gen_1 and work_gen_2):
                break

            # иначе рисуем
            draw_line_3D(data, iter_point_line1, iter_point_line2, color, thickness)

        # Если что-то осталось - заполнить точками из оставшейся линии
        # Если закончилась первая линия (Алгоритм в точке P2), но вторая ещё имеет точки на P1P3
        if work_gen_2:
            while True:
                setVoxel3D(data, iter_point_line2[0], iter_point_line2[1], iter_point_line2[2], color, thickness)
                # на всякий случай для отслеживания ошибочных ситуаций
                try:
                    iter_point_line2 = next(gen_line_2)
                except StopIteration as e:
                    print("Warning !!! Нет точек для оставшейся стороны 3-1 !")
                    break
                if iter_point_line2 == point1:
                    break

        # Если закончилась вторая линия (Алгоритм в точке P2), но первая ещё имеет точки на P2P3
        elif work_gen_1:
            while True:
                setVoxel3D(data, iter_point_line1[0], iter_point_line1[1], iter_point_line1[2], color, thickness)
                # на всякий случай для отслеживания ошибочных ситуаций
                try:
                    iter_point_line1 = next(gen_line_1)
                except StopIteration as e:
                    print("Warning !!! Нет точек для оставшейся стороны 2-1 !")
                    break
                if iter_point_line1 == point1:
                    break

        # дошли до конца
        return setVoxel3D(data, point1[0], point1[1], point1[2], color, thickness)
