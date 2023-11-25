from traceback import print_exc
from PIL import Image
from utils import convertToListImage, convertToPillowImage, saveImage
from services import getMean, getVariance, getStandardDeviation, getMSE, getRMSE, getPSNR, addGaussianAdditiveNoise

if __name__ == "__main__":
    try:
        with Image.open('./assets/cameraman.tif') as im:
            image = convertToListImage(im)

            mean = getMean(image)

            variance = getVariance(image, mean)

            standard_deviation = getStandardDeviation(image, mean, variance)

            print(f'Mean={mean}, variance={variance}, standard deviation={standard_deviation}.')
            
            std_dev_coef1 = 0.2

            noisy_image1 = addGaussianAdditiveNoise(image, std_dev_coef1)

            saveImage(convertToPillowImage(noisy_image1), f'./assets/cameraman_noisy_sigma={std_dev_coef1}.tif')
    except Exception as e:
        print('Error occured:')
        print_exc()