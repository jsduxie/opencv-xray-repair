# Image Processing Coursework
# James Duxbury BDZZ75

import sys
import os
import cv2
import math
import numpy as np
import re
from matplotlib import pyplot as plt

window_name_original = 'Input :('
window_name_transformed = "Output :)"

class file_handling:
    def __init__(self):
        self.status = 1
        if (len(sys.argv) != 2):
            print('Incorrect Argument Structure - should be in the format python main.py [directory]')
            self.status = -1
        else:
            if (not os.path.isdir(sys.argv[1])):
                print('Directory could not be found')
                self.status = -1
            else:
                self.directory = sys.argv[1]
                print(os.listdir(self.directory))
                self.unfiltered_images = os.listdir(self.directory)

                self.images = [file for file in self.unfiltered_images if re.match(r'.*\.(jpg|jpeg|png|gif)$', file)]
                self.images.sort()
                print(len(self.images), len(self.unfiltered_images))


        self.output_dir = "./Results"
    def save_image(self, image, filename):
        cv2.imwrite(f'{self.output_dir}/{filename}', image)



class image_processing:
    def __init__(self):
        image = cv2.imread(f'./{file.directory}/im001-healthy.jpg', cv2.IMREAD_COLOR)
        '''def click_event(event, x, y, flags, params):
            if event == cv2.EVENT_LBUTTONDOWN:
                print(f'({x}, {y})')
                cv2.putText(image, f'({x},{y})',(x,y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.circle(image, (x,y), 1, (255, 255, 255), 1)  

        
        
        pts_before = np.float32([[251, 229], [31, 238], [9,18], [234,8]])
        pts_after = np.float32([[255, 255], [0, 255], [0,0], [255,0]])
        M = cv2.getPerspectiveTransform(pts_before,pts_after)
        dst = cv2.warpPerspective(image, M,(255,255))
        cv2.imshow("Before", image)
        cv2.imshow("Warped", dst)
        

        cv2.namedWindow('Point Coordinates')

        cv2.setMouseCallback('Point Coordinates', click_event)

        while True:
            cv2.imshow('Point Coordinates',image)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("x"):
                break
        cv2.destroyAllWindows()'''
            
        
    def image_smoothing(self, image):
        #transformed = cv2.medianBlur(image, 3)
        transformed = cv2.bilateralFilter(image, d=9, sigmaColor=150, sigmaSpace=50)
        
        return transformed

        '''
        if not image is None:
            x = 1
            keep_processing = True
            while (keep_processing):
                transformed = cv2.medianBlur(image, 5)
                transformed2 = cv2.GaussianBlur(transformed, (0,0), x)
                

                cv2.imshow(window_name_original, image)
                cv2.imshow(window_name_transformed, transformed2)

                key = cv2.waitKey(0) # wait

                if (key == ord('x')):
                    cv2.destroyAllWindows() 
                    keep_processing = False
                elif (key == ord('v')):
                    x = x + 2
                elif (key == ord('c')):
                    c = c + 2
                elif (key == ord('a')):
                    alpha = alpha + 2'''
    def contrast_enhance(self, image, gamma):
        image_float = np.float32(image) / 255.0

        corrected_image_float = np.power(image_float, gamma)

        corrected_image_uint8 = np.uint8(corrected_image_float * 255)
        return corrected_image_uint8
    def brightness_adjust(self, image, amount):
        image_float = np.float32(image)
        brightened_image_float = image_float + amount
        brightened_image_float = np.clip(brightened_image_float, 0, 255)
        brightened_image_uint8 = np.uint8(brightened_image_float)
        return brightened_image_uint8
    def unwarped(self, image):
        pts_before = np.float32([[250, 227], [35, 233], [10,16], [233,6]])
        pts_after = np.float32([[255, 255], [0, 255], [0,0], [255,0]])
        M = cv2.getPerspectiveTransform(pts_before,pts_after)
        dst = cv2.warpPerspective(image, M,(255,255))
        return dst
    def denoise(self, image):
        dst = cv2.fastNlMeansDenoisingColored(image, h=9, hColor=9, templateWindowSize=7, searchWindowSize=21)
        return dst
    def histogram(self):
        img = cv2.imread('./Results/im001-healthy.jpg')
        assert img is not None, "file could not be read, check with os.path.exists()"
        color = ('b','g','r')
        for i,col in enumerate(color):
            histr = cv2.calcHist([img],[i],None,[256],[0,256])
            plt.plot(histr,color = col)
            plt.xlim([0,256])
        plt.show()
    def equalise(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        hsv[:,:,2] = cv2.equalizeHist(hsv[:,:,2])
        image = cv2.cvtColor(hsv, cv2.COLOR_YCrCb2BGR)

        '''b,g,r = cv2.split(image)
        b = cv2.equalizeHist(b) - 10
        #g = cv2.equalizeHist(g)
        r = cv2.equalizeHist(r)
        image = cv2.merge((b,g,r))'''

        return image
    def contrast_lab(self, image):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l,a,b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        new = cv2.merge((cl,a,b))
        result = cv2.cvtColor(new, cv2.COLOR_LAB2BGR)
        return result
    def sharpen(self, image):
        kernel = np.array([[0, -1, 0],[-1, 5, -1],[0, -1, 0]])
        result = cv2.filter2D(image, -1, kernel)
        return result
    def laplace(self, image):
        dst = cv2.Laplacian(image, cv2.CV_8U, ksize=3)
        image_new = cv2.subtract(image, dst)
        return image_new
    def adaptive_median_filter(self, image, max_kernel_size=9, threshold=[20,20,20]):
        # Define a function to calculate the median value within a given window
        def median_window(window):
            return np.median(window)

        # Apply adaptive median filtering to each pixel in each channel of the image
        filtered_image = np.copy(image)
        for c in range(image.shape[2]):  # Iterate over each channel
            channel = image[:, :, c]  # Extract the channel
            height, width = channel.shape[:2]
            for y in range(height):
                for x in range(width):
                    kernel_size = 3  # Initial kernel size
                    while kernel_size <= max_kernel_size:
                        # Define the local window centered around the current pixel
                        y_min = max(0, y - kernel_size // 2)
                        y_max = min(height - 1, y + kernel_size // 2)
                        x_min = max(0, x - kernel_size // 2)
                        x_max = min(width - 1, x + kernel_size // 2)
                        window = channel[y_min:y_max + 1, x_min:x_max + 1]

                        # Calculate the median value within the local window
                        median_value = median_window(window)

                        # If the difference between the median value and the pixel value is greater than the threshold,
                        # update the filtered image with the median value
                        if np.any(np.abs(median_value - channel[y, x]) > threshold[c]):
                            filtered_image[y, x, c] = median_value
                            break
                        else:
                            # If the difference is not significant, increase the kernel size
                            kernel_size += 2  # Increase kernel size by 2 (to keep it odd)

        return filtered_image
    def detect_contours(self, image):
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        height, width = image_gray.shape
        top_right = image_gray[0:height//2, width//2:width]
        
        edges = cv2.Canny(top_right, 400, 500)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        contour = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        radius = int(radius) + 3

        mask_full_size = np.zeros_like(image)
    
        cv2.circle(mask_full_size[0:height//2, width//2:width], center, radius, (255, 255, 255), -1)
        
        image_color = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
        
        overlay = cv2.addWeighted(image_color, 0.5, mask_full_size, 0.5, 0)

        #cv2.imshow("Overlay", overlay)
        
        #cv2.waitKey()
        return mask_full_size
    def inpaint(self, image, mask):
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        output = cv2.inpaint(image, mask, 5, cv2.INPAINT_TELEA)
        return output


    def process_image(self, filename):
        '''self.histogram()'''
        image = cv2.imread(f'./{file.directory}/{filename}', cv2.IMREAD_COLOR)

        
        image = self.adaptive_median_filter(image)

        image = self.unwarped(image)
        mask = self.detect_contours(image)
        image = self.inpaint(image, mask)
        
        
        image = self.denoise(image)
        image = self.sharpen(image)
        image = cv2.bilateralFilter(image, d=21, sigmaColor=21, sigmaSpace=21)

        image = self.laplace(image)
        image = self.equalise(image)
        
        
        image = self.denoise(image)
        
        
        return image
    def _calculate_average_cdf(self, reference_images: list):
        cumulative_sums_L = np.zeros((256,), dtype=np.float32)
        cumulative_sums_A = np.zeros((256,), dtype=np.float32)
        cumulative_sums_B = np.zeros((256,), dtype=np.float32)

        for reference in reference_images:
            reference_img = cv2.imread(reference, cv2.IMREAD_COLOR)
            reference_lab = cv2.cvtColor(reference_img, cv2.COLOR_BGR2LAB)
            hist_L = cv2.calcHist([reference_lab], [0], None, [256], [0, 256])
            hist_A = cv2.calcHist([reference_lab], [1], None, [256], [0, 256])
            hist_B = cv2.calcHist([reference_lab], [2], None, [256], [0, 256])

            cumulative_sums_L += np.cumsum(hist_L)
            cumulative_sums_A += np.cumsum(hist_A)
            cumulative_sums_B += np.cumsum(hist_B)

            num_images = len(reference_images)
            average_cdf_L = cumulative_sums_L / num_images
            average_cdf_A = cumulative_sums_A / num_images
            average_cdf_B = cumulative_sums_B / num_images
            average_cdf_L /= average_cdf_L[-1]
            average_cdf_A /= average_cdf_A[-1]
            average_cdf_B /= average_cdf_B[-1]

            self.average_cdf_L = average_cdf_L
            self.average_cdf_A = average_cdf_A
            self.average_cdf_B = average_cdf_B

    def _clean_reference_images(self, image):
        #references = ["./References/ref001.png", "./References/ref002.png", "./References/ref003.png", "./References/ref004.png", "./References/ref005.png", "./References/ref006.png", "./References/ref007.png"]
        average_cdf_L = self.average_cdf_L
        average_cdf_A = self.average_cdf_A
        average_cdf_B = self.average_cdf_B
        
        
        #, average_cdf_A, average_cdf_B = self._calculate_average_cdf(references)

        target_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        matched_lab = np.zeros_like(target_lab)

        target_hist_L = cv2.calcHist([target_lab], [0], None, [256], [0, 256])
        target_hist_A = cv2.calcHist([target_lab], [1], None, [256], [0, 256])
        target_hist_B = cv2.calcHist([target_lab], [2], None, [256], [0, 256])

        # Compute cumulative distribution functions (CDF) of the target image for each LAB channel
        target_cdf_L = np.cumsum(target_hist_L)
        target_cdf_A = np.cumsum(target_hist_A)
        target_cdf_B = np.cumsum(target_hist_B)

        # Normalize the CDFs to [0, 1] range
        target_cdf_L /= target_cdf_L[-1]
        target_cdf_A /= target_cdf_A[-1]
        target_cdf_B /= target_cdf_B[-1]

        # Initialize arrays to store the mapping functions for each LAB channel
        mapping_func_L = np.zeros(256, dtype=np.uint8)
        mapping_func_A = np.zeros(256, dtype=np.uint8)
        mapping_func_B = np.zeros(256, dtype=np.uint8)

        # Calculate the mapping functions for each gray level and channel
        for g1 in range(256):
            # Find the gray level in the target image that matches the CDF value of the reference image for each channel
            mapping_func_L[g1] = np.argmin(np.abs(target_cdf_L - average_cdf_L[g1]))
            mapping_func_A[g1] = np.argmin(np.abs(target_cdf_A - average_cdf_A[g1]))
            mapping_func_B[g1] = np.argmin(np.abs(target_cdf_B - average_cdf_B[g1]))

        # Apply the mapping functions to each channel of the target image
        matched_lab[:,:,0] = mapping_func_L[target_lab[:,:,0]]
        matched_lab[:,:,1] = mapping_func_A[target_lab[:,:,1]]
        matched_lab[:,:,2] = mapping_func_B[target_lab[:,:,2]]

        # Convert the matched LAB image back to BGR color space
        matched_img = cv2.cvtColor(matched_lab, cv2.COLOR_LAB2BGR)

        '''cv2.imshow("Damaged", image)
        cv2.imshow("Corrected", matched_img)
        cv2.waitKey()'''
        return matched_img




    def process_all_images(self):
        i = 1
        #self.histogram()
        references = ["./References/ref006.png"]
        self._calculate_average_cdf(references)
        for image in file.images:
            print(f"Processing image {i} of {len(file.images)} - {image}")
            i += 1
            print(image)
            self.image = image
            processed_image = self.process_image(image)
            file.save_image(processed_image, image)



            




file = file_handling()
if (file.status == -1):
    exit()


processing = image_processing()
#processing.process_image("im002-healthy.jpg")
processing.process_all_images()