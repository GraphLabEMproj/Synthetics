from Synthetic3D.src.hard.drawing_and_filliing.draw_element import setPixel2D

"""
setPixel(x,y) - функция закрашивает пиксель, с координатами x и y
"""

def get_line_point_2D(x1=0, y1=0, x2=0, y2=0):
    list_of_poinst = []

    dx = x2 - x1
    dy = y2 - y1

    sign_x = 1 if dx>0 else -1 if dx<0 else 0
    sign_y = 1 if dy>0 else -1 if dy<0 else 0

    if dx < 0: dx = -dx
    if dy < 0: dy = -dy

    if dx > dy:
        pdx, pdy = sign_x, 0
        es, el = dy, dx
    else:
        pdx, pdy = 0, sign_y
        es, el = dx, dy

    x, y = x1, y1

    error, t = el/2, 0
    list_of_poinst.append((x, y))

    while t < el:
        error -= es
        if error < 0:
            error += el
            x += sign_x
            y += sign_y
        else:
            x += pdx
            y += pdy
        t += 1
        list_of_poinst.append((x, y))
    return list_of_poinst

def draw_line_2D(img, x1=0, y1=0, x2=0, y2=0, color=255, thickness=1):
    if thickness<0:
        raise Exception("thickness не может быть меньше 0!")

    dx = x2 - x1
    dy = y2 - y1

    sign_x = 1 if dx>0 else -1 if dx<0 else 0
    sign_y = 1 if dy>0 else -1 if dy<0 else 0

    if dx < 0: dx = -dx
    if dy < 0: dy = -dy

    if dx > dy:
        pdx, pdy = sign_x, 0
        es, el = dy, dx
    else:
        pdx, pdy = 0, sign_y
        es, el = dx, dy

    x, y = x1, y1

    error, t = el/2, 0

    setPixel2D(img, x, y, color, thickness)

    while t < el:
        error -= es
        if error < 0:
            error += el
            x += sign_x
            y += sign_y
        else:
            x += pdx
            y += pdy
        t += 1
        setPixel2D(img, x, y, color, thickness)

    return img

def get_circle_point_2D(x, y, r):
    list_of_points = []
    disp_x = x
    disp_y = y
    x = 0
    y = r
    delta = (1-2*r)
    error = 0
    while y >= 0:
        list_of_points.append((disp_x + x, disp_y + y))
        list_of_points.append((disp_x + x, disp_y - y))
        list_of_points.append((disp_x - x, disp_y + y))
        list_of_points.append((disp_x - x, disp_y - y))

        error = 2 * (delta + y) - 1
        if ((delta < 0) and (error <=0)):
            x+=1
            delta = delta + (2*x+1)
            continue
        error = 2 * (delta - x) - 1
        if ((delta > 0) and (error > 0)):
            y -= 1
            delta = delta + (1 - 2 * y)
            continue
        x += 1
        delta = delta + (2 * (x - y))
        y -= 1
    return list_of_points

def draw_circle_2D(img, x, y, r, color, thickness):
    disp_x = x
    disp_y = y
    x = 0
    y = r
    delta = (1-2*r)
    error = 0
    while y >= 0:
        setPixel2D(img, disp_x + x, disp_y + y, color, thickness)
        setPixel2D(img, disp_x + x, disp_y - y, color, thickness)
        setPixel2D(img, disp_x - x, disp_y + y, color, thickness)
        setPixel2D(img, disp_x - x, disp_y - y, color, thickness)

        error = 2 * (delta + y) - 1
        if ((delta < 0) and (error <=0)):
            x+=1
            delta = delta + (2*x+1)
            continue
        error = 2 * (delta - x) - 1
        if ((delta > 0) and (error > 0)):
            y -= 1
            delta = delta + (1 - 2 * y)
            continue
        x += 1
        delta = delta + (2 * (x - y))
        y -= 1


def get_ellips_point_2D(x, y, r1, r2):
    list_of_points = []
    disp_x = x
    disp_y = y
    x = 0
    y = r2

    a2 = r1**2
    b2 = r2**2

    delta = a2*(3 - 2 * r2)

    while y >= 0:
        list_of_points.append((disp_x + x, disp_y + y))
        list_of_points.append((disp_x + x, disp_y - y))
        list_of_points.append((disp_x - x, disp_y + y))
        list_of_points.append((disp_x - x, disp_y - y))

        error = 2 * (delta + y) - 1
        if ((delta < 0) and (error <= 0)):
            x += 1
            delta = delta + b2 * (2 * x)
            continue
        y -= 1
        error = 2 * (delta - x) - 1
        if ((delta > 0) and (error > 0)):
            delta = delta - a2 * (2 * y)
            continue
        x += 1
        delta = delta + b2 * (2 * x) + a2 * (2 * y)

    return list_of_points

def draw_ellips_2D(img, x, y, r1, r2, color, thickness):
    disp_x = x
    disp_y = y
    x = 0
    y = r2

    a2 = r1**2
    b2 = r2**2

    delta = a2*(3 - 2 * r2)

    while y >= 0:
        setPixel2D(img, disp_x + x, disp_y + y, color, thickness)
        setPixel2D(img, disp_x + x, disp_y - y, color, thickness)
        setPixel2D(img, disp_x - x, disp_y + y, color, thickness)
        setPixel2D(img, disp_x - x, disp_y - y, color, thickness)

        error = 2 * (delta + y) - 1
        if ((delta < 0) and (error <= 0)):
            x += 1
            delta = delta + b2 * (2 * x)
            continue
        y -= 1
        error = 2 * (delta - x) - 1
        if ((delta > 0) and (error > 0)):
            delta = delta - a2 * (2 * y)
            continue
        x += 1
        delta = delta + b2 * (2 * x) + a2 * (2 * y)
