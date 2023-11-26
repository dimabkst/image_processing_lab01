from os.path import basename, splitext
from traceback import print_exc
from PIL import Image
from typing import List
from custom_types import FilterKernel
from constants import COMPUTED_DIRECTORY_NAME
from utils import validateImageSize, validateFilterKernel, convertToListImage, convertToPillowImage, saveImage
from services import getMean, getVariance, getStandardDeviation, getRMSE, getPSNR, addGaussianAdditiveNoise, createFilterKernel, linearSpatialFiltering

def labTask(image_path: str) -> None:
    computed_directory = f'./{COMPUTED_DIRECTORY_NAME}/'

    file_name_with_extension = basename(image_path)
    file_name, file_extension = splitext(file_name_with_extension)

    with Image.open(image_path) as im:
            image = convertToListImage(im)

            validateImageSize(image)

            mean = getMean(image)
            variance = getVariance(image, mean)
            standard_deviation = getStandardDeviation(image, mean, variance)

            print(f'Image = {image_path}, mean = {mean}, variance = {variance}, standard deviation = {standard_deviation}.')

            std_dev_coefs = [0.2, 0.3]

            noisy_images = [addGaussianAdditiveNoise(image, std_dev_coef) for std_dev_coef in std_dev_coefs]

            filter_kernels:List[FilterKernel] = [[[1, 2, 1], [2, 4, 2], [1, 2, 1]], [[1, 15, 1], [15, 30, 15], [1, 15, 1]]]
            filter_kernels = [createFilterKernel(filter_kernel) for filter_kernel in filter_kernels]
            
            for filter_kernel in filter_kernels:
                validateFilterKernel(filter_kernel)

            filtered_images = [[linearSpatialFiltering(noisy_image, filter_kernel) for noisy_image in noisy_images] for filter_kernel in filter_kernels]

            RMSEs = [[getRMSE(image, filtered_image) for filtered_image in _] for _ in filtered_images]
            PSNRs = [[getPSNR(image, filtered_images[i][j], RMSEs[i][j]) for j in range(len(filtered_images[i]))] for i in range(len(filtered_images))]

            for i in range(len(noisy_images)):
                saveImage(convertToPillowImage(noisy_images[i]), f'{computed_directory}{file_name}_noisy_{i + 1}{file_extension}')

            for i in range(len(filtered_images)):
                 for j in range(len(filtered_images[i])):
                    saveImage(convertToPillowImage(filtered_images[i][j]), f'{computed_directory}{file_name}_filtered_{i + 1}_{j + 1}{file_extension}')
            
            for i in range(len(RMSEs)):
                print_str = f'Kernel{i + 1}.'

                for j in range((len(RMSEs[i]))):
                    print_str += f' RMSE_{i + 1}_{j + 1} = {RMSEs[i][j]}. PSNR_{i + 1}_{j + 1} = {PSNRs[i][j]}.'

                print(print_str)

            print('\n')

if __name__ == "__main__":
    try:
         labTask('./assets/cameraman.tif') 

         labTask('./assets/lena_gray_256.tif')         
    except Exception as e:
        print('Error occured:')
        print_exc()