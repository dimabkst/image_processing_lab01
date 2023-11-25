from typing import Tuple
from custom_types import ListImage, FloatOrNone, FilterKernel, ImageFunction
from math import log10
from constants import MAX_NUMBER_OF_INTENSITY_LEVELS
from utils import generateGaussianNoise, convertToProperImage

def getMean(image: ListImage) -> float:
    N = len(image)
    M = len(image[0])

    mean = sum([sum([image[i][j] for j in range(M)]) for i in range(N)]) / (N * M)

    return mean

def getVariance(image: ListImage, mean: FloatOrNone=None) -> float:
    N = len(image)
    M = len(image[0])

    mean = mean if mean is not None else getMean(image)

    variance = sum([sum([(image[i][j] - mean) ** 2 for j in range(M)]) for i in range(N)]) / (N * M)

    return variance

def getStandardDeviation(image: ListImage, mean: FloatOrNone=None, variance: FloatOrNone=None) -> float:
    variance = variance if variance is not None else getVariance(image, mean)

    standard_deviation = variance ** 0.5

    return standard_deviation

def getMSE(image1: ListImage, image2: ListImage) -> float:
    N = len(image1)
    M = len(image1[0])

    MSE = sum([sum([(image1[i][j] - image2[i][j]) ** 2 for j in range(M)]) for i in range(N)]) / (N * M)

    return MSE

def getRMSE(image1: ListImage, image2: ListImage, MSE: FloatOrNone=None) -> float:
    MSE = MSE if MSE is not None else getMSE(image1, image2)
        
    RMSE = MSE ** 0.5

    return RMSE

def getPSNR(image1: ListImage, image2: ListImage, MSE: FloatOrNone=None, RMSE: FloatOrNone=None) -> float:
    L = MAX_NUMBER_OF_INTENSITY_LEVELS

    RMSE = RMSE if RMSE is not None else getRMSE(image1, image2, MSE)

    PSNR = (20 * log10((L - 1) / RMSE))

    return PSNR

def addGaussianAdditiveNoise(image: ListImage, std_dev_coef: float) -> ListImage:
    N = len(image)
    M = len(image[0])

    mean = getMean(image)

    standart_deviation = getStandardDeviation(image, mean)

    noise = generateGaussianNoise(mean, std_dev_coef * standart_deviation, (N, M))

    noisy_image = [[image[i][j] + noise[i][j] - mean for j in range(M)] for i in range(N)]

    return convertToProperImage(noisy_image)

def createFilterKernel(weights: FilterKernel) -> FilterKernel:
    filterKernel = weights

    filterKernelSum = sum([sum(filterKernel[row]) for row in range(len(filterKernel))])

    if filterKernelSum != 1:
        filterKernel = [[el / filterKernelSum for el in row] for row in filterKernel]

    return filterKernel

def getMirroredImageFunction(image: ListImage, filterKernelSizes: Tuple[int, int]) -> ImageFunction:
    N = len(image)
    M = len(image[0])
    
    extensionSizes = tuple(size // 2 for size in filterKernelSizes)

    def mirroredImageFunction(i: int, j: int) -> int:
        if i >= N + extensionSizes[0]:
            raise KeyError
        elif i < 0:
            ii = -i
        elif i >= N:
            ii = N - 1 - (i - N + 1)
        else:
            ii = i

        if j >= M + extensionSizes[1]:
            raise KeyError
        elif j < 0:
            jj = -j
        elif j >= M:
            jj = M - 1 - (j - M + 1)
        else:
            jj = j

        return image[ii][jj]

    return mirroredImageFunction

def getMirroredImage(image: ListImage, filterKernelSizes: Tuple[int, int]) -> ListImage:
    N = len(image)
    M = len(image[0])
    
    extensionSizes = tuple(size // 2 for size in filterKernelSizes)

    mirroredImage = []

    mirroredImageFunction = getMirroredImageFunction(image, filterKernelSizes)

    for i in range(-extensionSizes[0], N + extensionSizes[0]):
        mirroredImage.append([])

        for j in range(-extensionSizes[1], M + extensionSizes[1]):
            mirroredImage[-1].append(mirroredImageFunction(i, j))

    return mirroredImage

def linearSpatialFiltering(image: ListImage, filterKernel: FilterKernel) -> ListImage:
    N = len(image)
    M = len(image[0])

    filterKernelSizes = (len(filterKernel), len(filterKernel[0]))

    extendedImageFunction = getMirroredImageFunction(image, filterKernelSizes)

    a = filterKernelSizes[0] // 2 # equal to (filterKernelSizes[0] - 1) / 2 in formula
    b = filterKernelSizes[1] // 2

    filteredImage = [[sum([sum([filterKernel[a + s][b + t] * extendedImageFunction(i + s, j + t) for t in range(-b, b + 1)]) for s in range(-a, a + 1)]) for j in range(M)] for i in range(N)]

    return convertToProperImage(filteredImage)
