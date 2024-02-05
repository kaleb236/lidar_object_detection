import cv2 as cv
import numpy as np
import os
from time import time
from vision import Vision

# Change the working directory to the folder this script is in.
# Doing this because I'll be putting the files from each video in their own folder on GitHub
os.chdir(os.path.dirname(os.path.abspath(__file__)))


# initialize the WindowCapture class

# load the trained model
cascade_limestone = cv.CascadeClassifier('model/cascade.xml')
# load an empty Vision class
vision_limestone = Vision(None)

loop_time = time()
# while(True):

# get an updated image of the game
screenshot = cv.imread('trainer/positive/19.png')

# do object detection
rectangles = cascade_limestone.detectMultiScale(screenshot)
print(rectangles)

# draw the detection results onto the original image
detection_image = vision_limestone.draw_rectangles(screenshot, rectangles)

# display the images
cv.imshow('Matches', detection_image)

# debug the loop rate
print('FPS {}'.format(1 / (time() - loop_time)))
loop_time = time()

# press 'q' with the output window focused to exit.
# press 'f' to save screenshot as a positive image, press 'd' to 
# save as a negative image.
# waits 1 ms every loop to process key presses
key = cv.waitKey(0)
if key == ord('q'):
    cv.destroyAllWindows()
elif key == ord('f'):
    cv.imwrite('positive/{}.jpg'.format(loop_time), screenshot)
elif key == ord('d'):
    cv.imwrite('negative/{}.jpg'.format(loop_time), screenshot)

print('Done.')