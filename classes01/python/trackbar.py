import cv2

threshold_value=200
threshold_type=3
max_value=255
max_type=4
max_BINARY_value=255

windowName = "Threshold Demo"
trackbarType = "Type: \n 0: Binary \n 1: Binary Inverted \n 2: Truncate \n 3: To Zero \n 4: To Zero Inverted"
trackbarValue = "Value"

imGray = cv2.imread("../../data/images/threshold.png", cv2.IMREAD_GRAYSCALE)
cv2.namedWindow(windowName, cv2.WINDOW_AUTOSIZE)

def thresholdTypeDemo(*args):
    global threshold_type
    threshold_type = args[0]
    thresh, result = cv2.threshold(imGray, threshold_value, max_BINARY_value, threshold_type)
    cv2.imshow(windowName, result)


def thresholdValueDemo(*args):
    global threshold_value
    threshold_value = args[0]
    thresh, result = cv2.threshold(imGray, threshold_value, max_BINARY_value, threshold_type)
    cv2.imshow(windowName, result)

cv2.createTrackbar(trackbarType, windowName, threshold_type, max_type, thresholdTypeDemo)

cv2.createTrackbar(trackbarValue, windowName, threshold_value, max_value, thresholdValueDemo)

thresholdValueDemo(0)

# Wait until user finishes program
while True:
    c = cv2.waitKey(20)
    if c == 27:
        break

cv2.destroyAllWindows()