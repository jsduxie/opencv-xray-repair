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
        dst = cv2.fastNlMeansDenoisingColored(image, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21)
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
        #hsv[:,:,1] = cv2.equalizeHist(hsv[:,:,1])
        #hsv[:,:,0] = cv2.equalizeHist(hsv[:,:,0])
        image = cv2.cvtColor(hsv, cv2.COLOR_YCrCb2BGR)
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
    
    def adaptive_equalisation(self, image):
       
        image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(image_lab)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(15, 15))
        l_equalised = clahe.apply(l)
        a_equalised = clahe.apply(a)
        b_equalised = clahe.apply(b)
        clahe_image = cv2.merge((l_equalised, a, b))
        
        image = cv2.cvtColor(clahe_image, cv2.COLOR_LAB2BGR)
        return image
    def test(self, image):
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab_image)
        # Denoise the intensity (L) channel
        l_denoised = cv2.medianBlur(l, 7)
        kernel = np.array([[0, -1, 0],[-1, 5, -1],[0, -1, 0]])
        l_sharpened = cv2.filter2D(l_denoised, -1, kernel)

        # Denoise the color channels (A and B)
        a_denoised = cv2.bilateralFilter(a, 7, 150, 150)  
        a_sharpened = cv2.filter2D(a_denoised, -1, kernel)
        b_denoised = cv2.bilateralFilter(b, 7, 150, 150)  
        b_sharpened = cv2.filter2D(b_denoised, -1, kernel)

        # Merge the denoised channels back together
        denoised_lab_image = cv2.merge((l_denoised, a_sharpened, b_sharpened))

        # Convert the LAB denoised image back to BGR color space
        denoised_image = cv2.cvtColor(denoised_lab_image, cv2.COLOR_LAB2BGR)

        # Display the original and denoised images
        cv2.imshow('Original Image', image)
        cv2.imshow('Denoised Image', denoised_image)

        denoised_image = cv2.cvtColor(denoised_image, cv2.COLOR_BGR2YCrCb)
        dst = denoised_image
        dst = cv2.Laplacian(denoised_image, cv2.CV_8U, ksize=5)
        dst = cv2.divide(dst, 2)
        image = cv2.subtract(denoised_image, dst)
        image = cv2.cvtColor(denoised_image, cv2.COLOR_YCrCb2BGR)
        cv2.imshow('Laplaice Image', image)
        cv2.waitKey()


    def process_image(self, filename):
        image = cv2.imread(f'./{file.directory}/{filename}', cv2.IMREAD_COLOR)

        #image = self.contrast_enhance(image, 1.9)
        #image = cv2.medianBlur(image, 3)
        image = self.image_smoothing(image)
        image = self.unwarped(image)
        
        mask = self.detect_contours(image)
        image = self.inpaint(image, mask)
        
        #image = self.brightness_adjust(image, 50)
        #image = cv2.GaussianBlur(image,(3,3),1,1)
        image = cv2.bilateralFilter(image, 9, 150, 75)
        image = self.denoise(image)
        image = self.laplace(image)
        
        
        image = self.equalise(image)
        image = self.denoise(image)
        #image = self.sharpen(image)
        image = self.contrast_lab(image)

        return image
        

    def process_all_images(self):
        i = 1
        #self.histogram()
        
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
#processing.process_image("im001-healthy.jpg")
processing.process_all_images()



'''
def process_image(self, filename):
        image = cv2.imread(f'./{file.directory}/{filename}', cv2.IMREAD_COLOR)
        dst = cv2.dft(image.astype(np.float32), dst=None, flags=None, nonzeroRows=None)
        cv2.imshow("Fourier", dst)
        cv2.waitKey()
        #
        #image = self.contrast_enhance(image, 1.9)
        #image = cv2.medianBlur(image, 3)
        image = self.image_smoothing(image)
        image = self.unwarped(image)
        
        mask = self.detect_contours(image)
        image = self.inpaint(image, mask)
        
        #image = self.brightness_adjust(image, 50)
        #image = cv2.GaussianBlur(image,(3,3),1,1)
        #image = cv2.bilateralFilter(image, 9, 150, 75)
        image = self.denoise(image)
        image = self.laplace(image)
        
        
        image = self.equalise(image)
        image = self.denoise(image)
        #image = self.sharpen(image)
        image = self.contrast_lab(image)
        
        
        
        #image = self.contrast_enhance(image, 1.2)
        #image = self.image_smoothing(image)
        
        
        return image'''