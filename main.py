# Image Processing Coursework
# James Duxbury BDZZ75

import sys
import os
import cv2


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
        output_dir = "./outputs"

class image_processing:
    def __init__(self):
        print('Hello')     
        
        image = cv2.imread(f'./{file.directory}/im001-healthy.jpg', cv2.IMREAD_COLOR)
        if not image is None:

            keep_processing = True
            while (keep_processing):
                transformed = cv2.medianBlur(image, 5)
                cv2.imshow(window_name_original, image)
                cv2.imshow(window_name_transformed, transformed)
                key = cv2.waitKey(0) # wait

                if (key == ord('x')):
                    cv2.destroyAllWindows() 
                    keep_processing = False
                elif (key == ord('v')):
                    x = x + 2



            




file = file_handling()
if (file.status == -1):
    exit()

processing = image_processing()