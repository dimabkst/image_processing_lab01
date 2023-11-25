from typing import List, Tuple, Union
from numpy import array, clip, uint8
from numpy.random import normal
from PIL import Image
from constants import MIN_INTENSITY, MAX_INTENSITY

def validateImageSize(image: List[List[int]]):
    colsCount = len(image[0])

    for row in image:
        if (len(row) != colsCount):
            raise Exception('Image has inappropriate size')
        
def validateImagesSizeMatch(image1: List[List[int]], image2: List[List[int]]):
    if (len(image1) != len(image2)):
        raise Exception('Images have different rows count')
    
    if (len(image1[0]) != len(image2[0])):
        raise Exception('Images have different cols count')

def generateGaussianNoise(mean: float=0.0, std_dev: float=1.0, size:Union[Tuple[int, int], None]=None) -> List:
    return normal(mean, std_dev, size).tolist()

def convertToProperImage(image: List[List[float]]) -> List[List[int]]:
    # Convert values to int and limit them to min_intensity, max_intensity

    return clip(image, MIN_INTENSITY, MAX_INTENSITY).astype(uint8).tolist()

def convertToListImage(image: Image.Image) -> List[List[int]]:
    # In getpixel((x, y)) x - column, y - row 
    return [[image.getpixel((j, i)) for j in range(image.width)] for i in range(image.height)]

def convertToPillowImage(image: List[List[int]]) -> Image.Image:
    return Image.fromarray(array(image, dtype=uint8), mode='L')

def saveImage(image: Image.Image, path: str) -> None:
    return image.save(path, mode='L')