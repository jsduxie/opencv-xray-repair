James Duxbury, bdzz75
Image Processing Coursework

To run, please use the command python main.py [directory]
(Where directory is a path to the folder with the damaged images)

Execution takes around three hours on a Macbook Pro M1, but it can take up to five hours.

Inpainter.py is a helper script used solely for inpainting, and is imported into main.py. It needs to be present in the same directory as main.py (and run from this directory)

Inpainter.py is taken from https://github.com/NazminJuli/Criminisi-Inpainting by N. Nahar
and is an implementation of Criminisi's inpainting algorithm, which I modified slightly to work with my code.

workImage.jpg and updatedMask.jpg can be used to monitor the progress of inpainting while running

Required modules:
sys
os
time
math
cv2
numpy
re
