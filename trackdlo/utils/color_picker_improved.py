import cv2
import argparse
import numpy as np

# Improved version of color picker. Displays HSV image separated by channel both with and without thresholding applied. Additional visualization of the binary segmentation mask in color picker window. 
# Double clicking on a pixel in the "HSV separated" window prints out the channel value of that pixel.

def nothing(x):
    pass

ix,iy = -1,-1

# https://stackoverflow.com/questions/57401056/how-to-let-user-to-select-a-coordinate-in-an-image-and-take-those-coordinates-as
# stores mouse position in global variables ix(for x coordinate) and iy(for y coordinate) 
# on double click inside the image
def select_point(event,x,y,flags,param):
    global ix,iy
    if event == cv2.EVENT_LBUTTONDBLCLK: # captures left button double-click
        ix,iy = x,y

        if ix < 640:
            h = hsv[iy, ix, 0]
            print("hue:", h)

        elif ix < 640*2:
            s = hsv[iy, ix-640, 1]
            print("saturation:", s)
        else:
            v = hsv[iy, ix - 1280, 2]
            print("value:", v)


parser = argparse.ArgumentParser(description="Pick HSV color threshold")
parser.add_argument("path", help="Indicate /full-or-relative/path/to/image_file.png")
args = parser.parse_args()
img_path = f"{args.path}"

# Create a window
cv2.namedWindow('image')

# create trackbars for color change
cv2.createTrackbar('HMin','image',0,179,nothing) # Hue is from 0-179 for Opencv
cv2.createTrackbar('SMin','image',0,255,nothing)
cv2.createTrackbar('VMin','image',0,255,nothing)
cv2.createTrackbar('HMax','image',0,179,nothing)
cv2.createTrackbar('SMax','image',0,255,nothing)
cv2.createTrackbar('VMax','image',0,255,nothing)

# Set default value for MAX HSV trackbars.
cv2.setTrackbarPos('HMax', 'image', 179)
cv2.setTrackbarPos('SMax', 'image', 255)
cv2.setTrackbarPos('VMax', 'image', 255)

# Initialize to check if HSV min/max value changes
hMin = sMin = vMin = hMax = sMax = vMax = 0
phMin = psMin = pvMin = phMax = psMax = pvMax = 0


img = cv2.imread(img_path)
img = cv2.resize(img, (640, 480))
output = img
waitTime = 33

while(1):

    # get current positions of all trackbars
    hMin = cv2.getTrackbarPos('HMin','image')
    sMin = cv2.getTrackbarPos('SMin','image')
    vMin = cv2.getTrackbarPos('VMin','image')

    hMax = cv2.getTrackbarPos('HMax','image')
    sMax = cv2.getTrackbarPos('SMax','image')
    vMax = cv2.getTrackbarPos('VMax','image')

    # Set minimum and max HSV values to display
    lower = np.array([hMin, sMin, vMin])
    upper = np.array([hMax, sMax, vMax])

    # Create HSV Image and threshold into a range.
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    output = cv2.bitwise_and(img,img, mask= mask)

    # Print if there is a change in HSV value
    if( (phMin != hMin) | (psMin != sMin) | (pvMin != vMin) | (phMax != hMax) | (psMax != sMax) | (pvMax != vMax) ):
        print("(hMin = %d , sMin = %d, vMin = %d), (hMax = %d , sMax = %d, vMax = %d)" % (hMin , sMin , vMin, hMax, sMax , vMax))
        phMin = hMin
        psMin = sMin
        pvMin = vMin
        phMax = hMax
        psMax = sMax
        pvMax = vMax

    # Display output image
    img_stacked = cv2.vconcat([output, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)])
    cv2.imshow('image', img_stacked)

    # Display HSV separated
    hsv_concat = cv2.hconcat([hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]])

    cv2.namedWindow("HSV separated")
    cv2.setMouseCallback('HSV separated', select_point)
    cv2.imshow("HSV separated", hsv_concat)

    mask_h = cv2.inRange(hsv, (int(lower[0]), 0, 0), (int(upper[0]), 255, 255))
    mask_s = cv2.inRange(hsv, (0, int(lower[1]), 0), (179, int(upper[1]), 255))
    mask_v = cv2.inRange(hsv, (0, 0, int(lower[2])), (179, 255, int(upper[2])))

    mask_concat = cv2.hconcat([mask_h, mask_s, mask_v])

    hsv_masked_concat = cv2.bitwise_and(hsv_concat, hsv_concat, mask=mask_concat)
    cv2.imshow("Masked HSV separated", hsv_masked_concat)

    # Wait longer to prevent freeze for videos.
    if cv2.waitKey(waitTime) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()


