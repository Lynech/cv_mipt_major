import cv2
import numpy as np


def find_entrance_exit(image: np.ndarray) -> tuple:
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    nonzero = np.nonzero(image[0])[0]
    exit_l = nonzero[0]
    exit_r = nonzero[-1]
    inner_width = exit_r - exit_l + 1
    nonzero = np.nonzero(image[-1])[0]
    enter_l = nonzero[0]
    wall_width = np.nonzero(image[inner_width])[0][0]

    i = np.arange(
        wall_width + inner_width / 2,
        image.shape[0],
        wall_width + inner_width,
        dtype=int,
    )
    j = np.arange(
        wall_width + inner_width / 2,
        image.shape[-1],
        wall_width + inner_width,
        dtype=int,
    )
    exit_i = int(round((exit_l - wall_width) / (wall_width + inner_width)))
    enter_i = int(round((enter_l - wall_width) / (wall_width + inner_width)))

    radius = int(round(inner_width / 2)) + wall_width
    return (i, j, exit_i, enter_i, radius)


def color_left_side(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    x = np.nonzero(image[0])[0][0] - 1
    y = 0
    cv2.floodFill(image, None, (x, y), 100)
    return image


def accesable_points(image: np.ndarray, coodrs: list, curr: tuple, i, j) -> list:

    candidates = [
        (curr[0] - 1, curr[1]),
        (curr[0] + 1, curr[1]),
        (curr[0], curr[1] - 1),
        (curr[0], curr[1] + 1),
    ]
    if len(coodrs) == 0:
        candidates.remove((curr[0], curr[1] + 1))
    else:
        candidates.remove(coodrs[-1])
    accesable = []
    for path in candidates:
        if path[0] not in range(len(j)) or path[1] not in range(len(i)):
            continue
        else:
            a = min(i[curr[1]], i[path[1]])
            b = max(i[curr[1]], i[path[1]]) + 1
            c = min(j[curr[0]], j[path[0]])
            d = max(j[curr[0]], j[path[0]]) + 1
            seg = np.nonzero(image[a:b, c:d] - 255)
            if np.size(seg) == 0:
                accesable.append(path)
    return accesable


def make_decision(image: np.ndarray, accesable: list, radius: int, i, j) -> tuple:
    for path in accesable:
        if has_gray_n_black(image, path, radius, i, j):
            return path


def has_gray_n_black(image: np.ndarray, coord, radius, i, j):
    square = image[
        max(0, i[coord[1]] - radius) : i[coord[1]] + radius,
        j[coord[0]] - radius : min(j[coord[0]] + radius, image.shape[-1] - 1),
    ]
    return np.isin(0, square) and np.isin(100, square)


def find_way_from_maze(image: np.ndarray) -> tuple:
    """
    Найти путь через лабиринт.

    :param image: изображение лабиринта
    :return: координаты пути из лабиринта в виде (x, y), где x и y - это массивы координат
    """
    coords = []
    gray_image = image.copy()
    if gray_image.ndim == 3:
        gray_image = cv2.cvtColor(gray_image, cv2.COLOR_RGB2GRAY)

    i, j, exit_j, enter_j, radius = find_entrance_exit(gray_image)
    
    curr = (enter_j, len(i) - 1)
    end = (exit_j, 0)
    color_left_side(gray_image)

    while not coords or curr != end:
        accesable = accesable_points(gray_image, coords, curr, i, j)
        coords.append(curr)
        curr = make_decision(gray_image, accesable, radius, i, j)
    coords.append(curr)
    
    result = [(j[enter_j], image.shape[0])]
    for coord in coords:
        result.append((j[coord[0]], i[coord[1]]))
    result.append((j[exit_j], 0))

    return result
