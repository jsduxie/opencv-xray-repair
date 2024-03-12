# ===================================================================

# Example : save an image from file (and invert it)

# Author : Amir Atapour Abarghouei, amir.atapour-abarghouei@durham.ac.uk

# Copyright (c) 2024 Amir Atapour Abarghouei

# based on : https://github.com/tobybreckon/python-examples-ip/blob/master/skeleton.py
# License : LGPL - http://www.gnu.org/licenses/lgpl.html

# ===================================================================

import cv2

# ===================================================================

# read an image from the specified file (in colour)

img = cv2.imread('peppers.png', cv2.IMREAD_COLOR)

# check it has loaded

if not img is None:

    print("Processing and saving image.")

    # performing logical inversion - see OpenCV Manual entry for bitwise_not()

    inverted = cv2.bitwise_not(img)

    # write inverted image to file

    cv2.imwrite("inverted-peppers.png", inverted)

else:
    print("No image file was loaded.")

# ===================================================================
