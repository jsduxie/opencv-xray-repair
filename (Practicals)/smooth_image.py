# ===================================================================
# Example : reading, displaying and smoothing an image

# Author : Amir Atapour Abarghouei, amir.atapour-abarghouei@durham.ac.uk

# Copyright (c) 2024 Amir Atapour Abarghouei

# License : LGPL - http://www.gnu.org/licenses/lgpl.html
# ===================================================================
import cv2
# ===================================================================

# define display window name

windowName = "Original Image" # window name - for the original image
windowName2 = "Smoothed Image" # window name 2 - for the smoothed image

# read an image from the specified file - the cv2.IMREAD_COLOR flag enables reading the image in colour

image = cv2.imread('./peppers.png', cv2.IMREAD_COLOR)

# check that the image has been successfully loaded

if not image is None:

    # performing smoothing on the image using a 5x5 smoothing mark (see manual entry for GaussianBlur())
    x = 3
    blur = cv2.medianBlur(image, x)

    # display the original image and this blurred image in named windows

    cv2.imshow(windowName, image)
    cv2.imshow(windowName2, blur)

    # start the event loop - essential

    # cv2.waitKey() is a keyboard binding function (argument is the time in milliseconds).
    # It waits for specified milliseconds for any keyboard event.
    # If you press any key in that time, the program continues.
    # If 0 is passed, it waits indefinitely for a key stroke.

    key = cv2.waitKey(0) # wait

    # It can also be set to detect specific key strokes by recording which key is pressed

    # e.g. if user presses "x" then exit and close all windows

    if (key == ord('x')):
        cv2.destroyAllWindows()
    elif (key == ord("v")):
        x = x + 2
        blur = cv2.medianBlur(image, x)
        print('Hi')
else:
    print("No image file was loaded.")
# ===================================================================
