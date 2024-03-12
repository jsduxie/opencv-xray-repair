# ===================================================================
# Example : reading, displaying and interacting with an image

# Author : Amir Atapour Abarghouei, amir.atapour-abarghouei@durham.ac.uk

# Copyright (c) 2024 Amir Atapour Abarghouei

# License : LGPL - http://www.gnu.org/licenses/lgpl.html
# ===================================================================
import cv2
# ===================================================================

# mouse callback function - displays or sets image colour at the click
# location of the mouse

def colour_query_mouse_callback(event, x, y, flags, param):

    # records mouse events at postion (x,y) in the image window

    # left button click prints colour information at click location to stdout

    if event == cv2.EVENT_LBUTTONDOWN:
        # f-string is used here, which only works for python3 : https://realpython.com/python-f-strings/
        print(f"BGR colour at position ({x},{y}) = ({','.join(str(i) for i in image[y,x])})")

    # right button sets colour information at click location to white

    elif event == cv2.EVENT_RBUTTONDOWN:
        image[y,x] = [255,255,255]

# ===================================================================

# define display window name

windowName = "Displayed Image" # window name - for the displayed image

# read an image from the specified file - the cv2.IMREAD_COLOR flag enables reading the image in colour

image = cv2.imread('./peppers.png', cv2.IMREAD_COLOR)

# check that the image has been successfully loaded

if not image is None:

    # create a named window object

    cv2.namedWindow(windowName)

    # set the mouse call back function that will be called every time
    # the mouse is clicked inside the associated window

    cv2.setMouseCallback(windowName,colour_query_mouse_callback)

    # set a loop control flag

    keep_processing = True

    while (keep_processing):

        # display this blurred image in a named window

        cv2.imshow(windowName, image)

        # start the event loop - essential

        # cv2.waitKey() is a keyboard binding function (argument is the time in milliseconds).
        # It waits for specified milliseconds for any keyboard event.
        # If you press any key in that time, the program continues.
        # If 0 is passed, it waits indefinitely for a key stroke.
        # (bitwise and with 0xFF to extract least significant byte of multi-byte response)

        key = cv2.waitKey(40) & 0xFF # wait 40ms (i.e. 1000ms / 25 fps = 40 ms)

        # It can also be set to detect specific key strokes by recording which key is pressed

        # e.g. if user presses "x" then exit

        if (key == ord('x')):
            keep_processing = False

else:
    print("No image file was loaded.")

# ... and now close all windows

cv2.destroyAllWindows()
# ===================================================================
