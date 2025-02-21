import cv2
import numpy as np
import math
import matplotlib.pyplot as plt


def rotate(image, point: tuple, angle: float) -> np.ndarray:
    """
    Повернуть изображение по часовой стрелке на угол от 0 до 360 градусов и преобразовать размер изображения.

    :param image: исходное изображение
    :param point: значение точки (x, y), вокруг которой повернуть изображение
    :param angle: угол поворота
    :return: повернутное изображение
    """
    y, x, _ = image.shape
    cos = abs(math.cos(angle / 180 * math.pi))
    sin = abs(math.sin(angle / 180 * math.pi))
    X = int(y * sin + x * cos)
    Y = int(y * cos + x * sin)
    M0 = cv2.getRotationMatrix2D((0, 0), angle, 1)[0:2, 0:2]
    angels = np.array([[0, 0], [x, 0], [0, y], [x, y]]).T
    shifted = np.matmul(M0, angels)
    shift = -np.min(shifted, axis=1).reshape((2, -1))
    M1 = np.block([cv2.getRotationMatrix2D(shift.flatten(), angle, 1)[:, 0:2], shift])
    image = cv2.warpAffine(image, M1, (X, Y))
    return image


def scan_image(image: np.ndarray):
    points1, points2 = find_points(image)
    image = apply_warpAffine(image, points1, points2)

    return image


def find_edges(image: np.ndarray) -> list:
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    mask1 = cv2.inRange(image, (0, 100, 0), (5, 255, 255))
    mask2 = cv2.inRange(image, (170, 100, 0), (180, 255, 255))
    mask = cv2.bitwise_or(mask1, mask2)
    image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
    nm = np.nonzero(mask)
    nm2 = np.nonzero(mask.T)
    result = np.float32(
        [
            (nm[1][0], nm[0][0]),
            (nm2[0][-1], nm2[1][-1]),
            (nm2[0][0], nm2[1][0]),
            (nm[1][-1], nm[0][-1]),
        ]
    )
    return result


def find_points(image):
    points1 = find_edges(image)
    rows, cols, _ = image.shape

    u0 = (cols) / 2.0
    v0 = (rows) / 2.0

    w1 = np.linalg.norm(points1[0] - points1[1])
    w2 = np.linalg.norm(points1[2] - points1[3])

    h1 = np.linalg.norm(points1[0] - points1[2])
    h2 = np.linalg.norm(points1[1] - points1[3])

    w = max(w1, w2)
    h = max(h1, h2)

    ar_vis = float(w) / float(h)

    m1 = np.array((points1[0][0], points1[0][1], 1)).astype("float32")
    m2 = np.array((points1[1][0], points1[1][1], 1)).astype("float32")
    m3 = np.array((points1[2][0], points1[2][1], 1)).astype("float32")
    m4 = np.array((points1[3][0], points1[3][1], 1)).astype("float32")

    k2 = np.dot(np.cross(m1, m4), m3) / np.dot(np.cross(m2, m4), m3)
    k3 = np.dot(np.cross(m1, m4), m2) / np.dot(np.cross(m3, m4), m2)

    n2 = k2 * m2 - m1
    n3 = k3 * m3 - m1

    n21 = n2[0]
    n22 = n2[1]
    n23 = n2[2]

    n31 = n3[0]
    n32 = n3[1]
    n33 = n3[2]

    f = math.sqrt(
        np.abs(
            (1.0 / (n23 * n33))
            * (
                (n21 * n31 - (n21 * n33 + n23 * n31) * u0 + n23 * n33 * u0 * u0)
                + (n22 * n32 - (n22 * n33 + n23 * n32) * v0 + n23 * n33 * v0 * v0)
            )
        )
    )

    A = np.array([[f, 0, u0], [0, f, v0], [0, 0, 1]]).astype("float32")

    At = np.transpose(A)
    Ati = np.linalg.inv(At)
    Ai = np.linalg.inv(A)

    ar_real = math.sqrt(
        np.dot(np.dot(np.dot(n2, Ati), Ai), n2)
        / np.dot(np.dot(np.dot(n3, Ati), Ai), n3)
    )

    if ar_real < ar_vis:
        W = int(w)
        H = int(W / ar_real)
    else:
        H = int(h)
        W = int(ar_real * H)

    points2 = np.float32([[0, 0], [W, 0], [0, H], [W, H]])
    return (points1, points2)


def apply_warpAffine(image, points1, points2) -> np.ndarray:
    """
    Применить афинное преобразование согласно переходу точек points1 -> points2 и
    преобразовать размер изображения.

    :param image:
    :param points1:
    :param points2:
    :return: преобразованное изображение
    """
    M = cv2.getPerspectiveTransform(points1, points2)
    image = cv2.warpPerspective(image, M, (int(points2[-1, 0]), int(points2[-1, 1])))
    return image
