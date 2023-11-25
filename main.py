from traceback import print_exc
from PIL import Image
from utils import validateImageSize, validateFilterKernel, convertToListImage, convertToPillowImage, saveImage
from services import getMean, getVariance, getStandardDeviation, getMSE, getRMSE, getPSNR, addGaussianAdditiveNoise, getMirroredImage, linearSpatialFiltering

if __name__ == "__main__":
    try:
        with Image.open('./assets/cameraman.tif') as im:
            image = convertToListImage(im)

            validateImageSize(image)

            mean = getMean(image)

            variance = getVariance(image, mean)

            standard_deviation = getStandardDeviation(image, mean, variance)

            print(f'Mean={mean}, variance={variance}, standard deviation={standard_deviation}.')

            std_dev_coef1 = 0.2

            noisy_image1 = addGaussianAdditiveNoise(image, std_dev_coef1)

            saveImage(convertToPillowImage(noisy_image1), f'./assets/cameraman_noisy_sigma={std_dev_coef1}.tif')

            filter_kernel_size = 250

            mirroredImage = getMirroredImage(image, (filter_kernel_size, filter_kernel_size))

            saveImage(convertToPillowImage(mirroredImage), f'./assets/cameraman_mirrored_size={filter_kernel_size}.tif')

            filterKernel = [[1, 2, 1], [2, 4, 2], [1, 2, 1]]

            filterKernelSum = sum([sum(filterKernel[row]) for row in range(len(filterKernel))])

            filterKernel = [[el / filterKernelSum for el in row] for row in filterKernel]

            validateFilterKernel(filterKernel)

            filteredImage = linearSpatialFiltering(noisy_image1, filterKernel)

            saveImage(convertToPillowImage(filteredImage), f'./assets/cameraman_filtered.tif')            
    except Exception as e:
        print('Error occured:')
        print_exc()