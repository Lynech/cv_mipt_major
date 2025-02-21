import cv2
import numpy as np
import matplotlib.pyplot as plt


def find_lines(image: np.ndarray) -> np.array:
    mask = 255 - cv2.inRange(image, (20, 50, 150), (30, 150, 255))
    widths = np.array(np.nonzero(np.diff(mask[0]))).reshape((-1, 2))
    return widths


def empty_road(image: np.ndarray, widths: np.array) -> int:
    mask = cv2.inRange(image, (0, 100, 100), (10, 255, 255))
    road = -1
    for i, w in enumerate(widths):
        if mask[:, w[0] : w[1]].max() == 0:
            road = i
    return road


def is_curr_road(image: np.ndarray, width: np.array) -> int:
    mask = cv2.inRange(image, (100, 100, 100), (140, 255, 255))
    if mask[:, width[0] : width[1]].max() != 0:
        return True
    return False


def find_road_number(image: np.ndarray) -> int:
    """
    Найти номер дороги, на которой нет препятсвия в конце пути.

    :param image: исходное изображение
    :return: номер дороги, на котором нет препятсвия на дороге
    """
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    widths = find_lines(image)
    road_number = empty_road(image, widths)
    if road_number == -1 or is_curr_road(image, widths[road_number]):
        return None
    
    return road_number
