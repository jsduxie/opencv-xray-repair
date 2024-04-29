# Image Processing Coursework
# James Duxbury BDZZ75

'''
Execution time takes just under 2.5 hours to process all images
on a MacBook Pro M1, 8GB RAM. An alternative, Windows, computer,
takes around 5 hours.

Please allow sufficient time to run this :)
'''

# File Imports
import sys
import os
import re
import numpy as np
import cv2

# Modified Criminisi inpainter found in Inpainter.py
# https://github.com/NazminJuli/Criminisi-Inpainting
from Inpainter import Inpainter


# FileHandling Class
class FileHandling:
    '''
    This class is used to handle argument parsing and image saving
    '''
    def __init__(self):
        # This function parses arguments, and stores a list of all images

        # Status is used to flag errors with argument parsing
        self.status = 1
        if len(sys.argv) != 2:  # Incorrect amount of arguments
            print('Arguments should be in format python main.py [directory]')
            self.status = -1
        else:  # Directory could not be found
            if not os.path.isdir(sys.argv[1]):
                print('Directory could not be found')
                self.status = -1
            else:
                self.directory = sys.argv[1]

                # Find list of all files within specified folder
                self.unfiltered_images = os.listdir(self.directory)

                # To prevent errors, remove any non .jpg files
                self.images = [
                    file for file in self.unfiltered_images
                    if re.match(r'.*\.(jpg)$', file)
                    ]

                # Sort images so they are processed in order
                self.images.sort()

        # Sets the output directory to save processed images
        # Makes directory if it doesn't exist
        if not os.path.exists('./Results'):
            os.makedirs('./Results')
        self.output_dir = "./Results"

    def save_image(self, image, filename):
        # Save each repaired image when needed
        cv2.imwrite(f'{self.output_dir}/{filename}', image)

# Class to handle image processing
class ImageProcessing:
    '''
    This class contains methods to process the x-ray images
    '''
    def perspective_transform(self, image):
        # All images had same warping, so I found these coordinates
        # of the corners in pts_before
        pts_before = np.float32([[250, 227], [35, 233], [10, 16], [233, 6]])
        pts_after = np.float32([[255, 255], [0, 255], [0, 0], [255, 0]])

        # M is the transformation matrix M pts_before => pts_after
        m = cv2.getPerspectiveTransform(pts_before, pts_after)

        # Apply to image
        dst = cv2.warpPerspective(image, m, (255,255))
        return dst

    def denoise(self, image, h=10, hColor=10):
        # This function uses OpenCV's implementation of Non-Local Means denoiser to remove noise
        dst = cv2.fastNlMeansDenoisingColored(image, h=h, hColor=hColor,
                                              templateWindowSize=7, searchWindowSize=21)
        return dst

    def histogram_equalisation(self, image):
        # This function performs Histogram Equalisation on the Cb channel in YCrCb colour space
        # Convert image from BGR to YCrCb
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

        # Equalise Cb channel
        ycrcb[:, :, 2] = cv2.equalizeHist(ycrcb[:, :, 2])

        # Convert back to BGR for further processing, and return
        image = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        return image

    def contrast_lab(self, image):
        # This function applies contrast enhancement on the Lightness channel in LAB colour space
        # Convert image from BGR to LAB
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        # Split image into separate channels
        l,a,b = cv2.split(lab)

        # Create and apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l)

        # Merge channels, convert to BGR for further processing, and return
        result = cv2.merge((cl, a, b))
        result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
        return result

    def laplace(self, image, ksize=3):
        # This function uses the Laplace Operator to sharpen the edges of the images
        # Apply the Laplacian Operator to the image
        laplace = cv2.Laplacian(image, cv2.CV_8U, ksize=ksize)

        # Subtract this from the original image, and return
        image_new = cv2.subtract(image, laplace)
        return image_new

    def detect_contours(self, image):
        # This function detects mask using Canny edge detection
        # Convert to greyscale
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Extract region of interest (ROI), which is the top-right in this case
        height, width = image_gray.shape
        top_right = image_gray[0:height//2, width//2:width]

        # Perform Canny edge detection
        edges = cv2.Canny(top_right, 400, 500) #400,500


        # Produce contours from the edges, and isolate the largest contour
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea)

        # Create a minimum enclosing circle for the contour, slightly larger to allow for coverage
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        radius = int(radius) + 3

        # Initialise mask, we want this to be zeros everywhere except the circle
        mask_full_size = np.zeros_like(image_gray)

        # Add minimum enclosing circle to the mask, and return
        cv2.circle(mask_full_size[0:height//2, width//2:width], center, radius, 255, -1)
        return mask_full_size

    def inpaint_opencv(self, image, mask):
        # Deprecated, this uses the OpenCV inpainting function, I now use a modified
        # Criminisi implementation Inpainter.py https://github.com/NazminJuli/Criminisi-Inpainting
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        output = cv2.inpaint(image, mask, 5, cv2.INPAINT_TELEA)
        return output

    def adaptive_median_filter(self, image, max_kernel_size=13, threshold=[10, 10, 10]):
        # This function is my implementation of the adaptive median filter

        # Helper function to return the median of a neighbourhood
        def neighbourhood_median(neighbourhood):
            return np.median(neighbourhood)

        # Initialise result (filtered) image
        filtered_image = np.copy(image)

        # Iterate over every channel in the image
        for c in range(image.shape[2]):
            channel = image[:, :, c]
            height, width = channel.shape[:2]

            # Iterate through each pixel in the image (for this channel)
            for y in range(height):
                for x in range(width):
                    # Initialise kernel_size
                    kernel_size = 3

                    while kernel_size <= max_kernel_size:
                        # Locate the pixel's neighbourhood
                        y_min = max(0, y - kernel_size // 2)
                        y_max = min(height - 1, y + kernel_size // 2)
                        x_min = max(0, x - kernel_size // 2)
                        x_max = min(width - 1, x + kernel_size // 2)
                        neighbourhood = channel[y_min:y_max + 1, x_min:x_max + 1]

                        # Calculate the neighbourhood's median value
                        median_value = neighbourhood_median(neighbourhood)

                        # Update pixel's value if and only if it differs significantly
                        if np.any(np.abs(median_value - channel[y, x]) > threshold[c]):
                            filtered_image[y, x, c] = median_value
                            break
                        else:
                            # Increase the kernel_size by 2 if difference is insignificant
                            kernel_size += 2

        return filtered_image

    def sobel_edge_enhance(self, image):
        # Sobel edge enhancement
        # Calculate horizontal and vertical gradients
        sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

        # Convert gradients to positive integers
        sobel_x = cv2.convertScaleAbs(sobel_x)
        sobel_y = cv2.convertScaleAbs(sobel_y)

        # Combine gradients and overlay on image
        sobel = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)
        result = cv2.addWeighted(image, 1, sobel, 0.1, 0)
        return result

    def colour_scaling(self, image):
        # Final scaling of colours
        # Split colour channels
        b, g, r = cv2.split(image)

        # Scale green and red channels
        g = np.clip(g * 1.1, 0, 255).astype(np.uint8)
        r = np.clip(r * 0.9, 0, 255).astype(np.uint8)

        result = cv2.merge((b, g, r))
        return result

    def process_image(self, filename):
        # This function is used to fully process each individual image using the above functions

        # Read in image
        image = cv2.imread(f'./{file.directory}/{filename}', cv2.IMREAD_COLOR)

        # Perform adaptive median filter
        print("Applying Adaptive Median Filter...")
        image = self.adaptive_median_filter(image, threshold=[7,7,7])

        # Perform Projective Transform
        print("Applying Projective Transform...")
        image = self.perspective_transform(image)

        # Determine mask for inpainting
        print('Determining Mask...')
        mask = self.detect_contours(image)

        # Inpaint missing region
        # THIS USES AN IMPLEMENTATION OF CRIMINISI'S ALGORITHM
        # FROM https://github.com/NazminJuli/Criminisi-Inpainting
        # MODIFIED CODE CAN BE FOUND IN Inpainter.py
        print("Inpainting. Please Wait...")
        i = Inpainter(image, mask, 4)

        if i.checkValidInputs()== i.CHECK_VALID:
            i.inpaint()
            image = i.result
            print("Inpainting Complete")
        else:
            print('Unable to inpaint, invalid inputs')

        # Apply Non-Local Means Denoiser
        print("First Application of Non-Local Means...")
        image = self.denoise(image, 12, 12)

        # Apply Laplacian Transform
        print("Conducting Laplacian Transform...")
        image = self.laplace(image)

        # Apply Histogram Equalisation in YCrCb colour space
        print("Histogram Equalisation...")
        image = self.histogram_equalisation(image)

        # Reapply Non-Local Means Denoiser, slightly weaker
        print("Second Application of Non-Local Means...")
        image = self.denoise(image, 10, 10)

        # Contrast Enhancement in LAB colour space
        print("Applying CLAHE...")
        image = self.contrast_lab(image)

        # Sobel Edge Enhancement
        print("Applying Sobel Edge Enhancement...")
        image = self.sobel_edge_enhance(image)

        # Colour Scaling
        print("Scaling Colours...")
        image = self.colour_scaling(image)

        # Return image
        print("Image Processed.")
        return image

    def process_all_images(self):
        # This function is used to iterate through all images, process them, and save them
        # i is a counter to keep track of image progress for output logging.
        i = 1
        for image in file.images:
            print(f"\n-------------------\nProcessing image {i} of {len(file.images)} - {image}")
            i += 1

            # Process the image
            processed_image = self.process_image(image)

            # Save image in predefined directory
            file.save_image(processed_image, image)


# Handle arguments and read in file names
file = FileHandling()
if file.status == -1:
    sys.exit()

# Initialise class, and then process images
processing = ImageProcessing()
processing.process_all_images()
