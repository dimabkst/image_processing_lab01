from os.path import splitext
from traceback import print_exc
from PIL import Image
from utils import validateImageSize, validateFilterKernel, convertToListImage, convertToPillowImage, saveImage
from services import getMean, getVariance, getStandardDeviation, getRMSE, getPSNR, addGaussianAdditiveNoise, createFilterKernel, linearSpatialFiltering

def labTask(imagePath: str) -> None:
    path_without_extension, path_extension = splitext(imagePath)

    with Image.open(imagePath) as im:
            image = convertToListImage(im)

            validateImageSize(image)

            mean = getMean(image)
            variance = getVariance(image, mean)
            standard_deviation = getStandardDeviation(image, mean, variance)

            print(f'Image = {imagePath}, mean = {mean}, variance = {variance}, standard deviation = {standard_deviation}.')

            # TODO: create lists and not variables
            std_dev_coef1 = 0.2
            std_dev_coef2 = 0.3

            noisy_image1 = addGaussianAdditiveNoise(image, std_dev_coef1)
            noisy_image2 = addGaussianAdditiveNoise(image, std_dev_coef2)

            filterKernel1 = createFilterKernel([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
            filterKernel2 = createFilterKernel([[1, 15, 1], [15, 30, 15], [1, 15, 1]])            

            validateFilterKernel(filterKernel1)
            validateFilterKernel(filterKernel2)

            filteredImage1_1 = linearSpatialFiltering(noisy_image1, filterKernel1)
            filteredImage2_1 = linearSpatialFiltering(noisy_image2, filterKernel1)

            filteredImage1_2 = linearSpatialFiltering(noisy_image1, filterKernel2)
            filteredImage2_2 = linearSpatialFiltering(noisy_image2, filterKernel2)

            RMSE1_1 = getRMSE(image, filteredImage1_1)
            PSNR1_1 = getPSNR(image, filteredImage1_1, RMSE=RMSE1_1)
            RMSE2_1 = getRMSE(image, filteredImage2_1)
            PSNR2_1 = getPSNR(image, filteredImage2_1, RMSE=RMSE2_1)

            RMSE1_2 = getRMSE(image, filteredImage1_2)
            PSNR1_2 = getPSNR(image, filteredImage1_2, RMSE=RMSE1_2)
            RMSE2_2 = getRMSE(image, filteredImage2_2)
            PSNR2_2 = getPSNR(image, filteredImage2_2, RMSE=RMSE2_2)

            saveImage(convertToPillowImage(noisy_image1), f'{path_without_extension}_noisy_1{path_extension}')
            saveImage(convertToPillowImage(noisy_image2), f'{path_without_extension}_noisy_2{path_extension}')
            saveImage(convertToPillowImage(filteredImage1_1), f'{path_without_extension}_filtered_1_1{path_extension}')
            saveImage(convertToPillowImage(filteredImage2_1), f'{path_without_extension}_filtered_2_1{path_extension}')
            saveImage(convertToPillowImage(filteredImage1_2), f'{path_without_extension}_filtered_1_2{path_extension}')
            saveImage(convertToPillowImage(filteredImage2_2), f'{path_without_extension}_filtered_2_2{path_extension}')

            print(f'Kernel1. RMSE1_1: {RMSE1_1}. PSNR1_1: {PSNR1_1}. RMSE2_1: {RMSE2_1}. PSNR2_1: {PSNR2_1}')
            print(f'Kernel2. RMSE1_2: {RMSE1_2}. PSNR1_2: {PSNR1_2}. RMSE2_2: {RMSE2_2}. PSNR2_2: {PSNR2_2}')


if __name__ == "__main__":
    try:
         labTask('./assets/cameraman.tif')          
    except Exception as e:
        print('Error occured:')
        print_exc()