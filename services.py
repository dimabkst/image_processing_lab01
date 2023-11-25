from typing import List, Tuple, Union
from math import log10, floor 
from constants import MAX_NUMBER_OF_INTENSITY_LEVELS
from utils import generateGaussianNoise, convertToProperImage

def getMean(image: List[List[int]]) -> float:
    N = len(image)
    M = len(image[0])

    mean = sum([sum([image[i][j] for j in range(M)]) for i in range(N)]) / (N * M)

    return mean

def getVariance(image: List[List[int]], mean: Union[float, None]=None) -> float:
    N = len(image)
    M = len(image[0])

    mean = mean if mean is not None else getMean(image)

    variance = sum([sum([(image[i][j] - mean) ** 2 for j in range(M)]) for i in range(N)]) / (N * M)

    return variance

def getStandardDeviation(image: List[List[int]], mean: Union[float, None]=None, variance: Union[float, None]=None) -> float:
    variance = variance if variance is not None else getVariance(image, mean)

    standard_deviation = variance ** 0.5

    return standard_deviation

def getMSE(image1: List[List[int]], image2: List[List[int]]) -> float:
    N = len(image1)
    M = len(image1[0])

    MSE = sum([sum([(image1[i][j] - image2[i][j]) ** 2 for j in range(M)]) for i in range(N)]) / (N * M)

    return MSE

def getRMSE(image1: List[List[int]], image2: List[List[int]], MSE: Union[float, None]=None) -> float:
    MSE = MSE if MSE is not None else getMSE(image1, image2)
        
    RMSE = MSE ** 0.5

    return RMSE

def getPSNR(image1: List[List[int]], image2: List[List[int]], MSE: Union[float, None]=None, RMSE: Union[float, None]=None) -> float:
    L = MAX_NUMBER_OF_INTENSITY_LEVELS

    RMSE = RMSE if RMSE is not None else getRMSE(image1, image2, MSE)

    PSNR = (20 * log10((L - 1) / RMSE))

    return PSNR

def addGaussianAdditiveNoise(image: List[List[int]], std_dev_coef: float) -> List[List[int]]:
    N = len(image)
    M = len(image[0])

    mean = getMean(image)

    standart_deviation = getStandardDeviation(image, mean)

    noise = generateGaussianNoise(mean, std_dev_coef * standart_deviation, (N, M))

    noisy_image = [[image[i][j] + noise[i][j] - mean for j in range(M)] for i in range(N)]

    return convertToProperImage(noisy_image)

def mirrorImage(image: List[List[int]], filterKernelSizes: Tuple[int, int]) -> List[List[int]]:
    N = len(image)
    M = len(image[0])
    
    extensionSizes = tuple(floor(size / 2) for size in filterKernelSizes)

    mirroredImage = []

    for i in range(-extensionSizes[0], N + extensionSizes[0]):
        mirroredImage.append([])

        if i < 0:
            ii = -i
        elif i >= N:
            ii = N - 1 - (i - N + 1)
        else:
            ii = i

        for j in range(-extensionSizes[1], M + extensionSizes[1]):
            if j < 0:
                jj = -j
            elif j >= M:
                jj = M - 1 - (j - M + 1)
            else:
                jj = j

            # if i < 0 and j < 0:
            #     pixel = image[-i][-j]
            # elif i < 0:
            #     pixel = image[-i][-j]
            # elif j < 0:
            #     pixel = image[i][-j]
            # elif i >= N and j >= M:
            #     pixel = image[N - 1 - (i - N + 1)][M - 1 - (j - M + 1)]
            # elif i >= N:
            #     pixel = image[N - 1 - (i - N + 1)][j]
            # elif i >= M:
            #     pixel = image[i][M - 1 - (j - M + 1)]
            # else:
            #     pixel = image[i][j]

            mirroredImage[-1].append(image[ii][jj])

    return mirroredImage