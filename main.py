import cv2 as cv
from time import sleep
import numpy as np

cv.namedWindow("Parameters")
cv.resizeWindow("Parameters", 800, 600)
cv.createTrackbar("Threshold1", "Parameters", 68, 255, lambda _: None)
cv.createTrackbar("Threshold2", "Parameters", 130, 255, lambda _: None)
cv.createTrackbar("DilationIterations", "Parameters", 1, 10, lambda _: None)
cv.createTrackbar("KernelSize", "Parameters", 1, 10, lambda _: None)
cv.createTrackbar("Min Area", "Parameters", 1, 1000, lambda _: None)
cv.createTrackbar("LineThreshold", "Parameters", 400, 500, lambda _: None)

img = cv.imread("images/input/stock-photo.png", cv.IMREAD_GRAYSCALE)
img = cv.resize(img, (0, 0), fx=0.75, fy=0.75)

imgBlur = cv.GaussianBlur(img, (7, 7), 1)

# images.append(cv.cvtColor(images[-1], cv.COLOR_BGR2GRAY))

def ccw(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

# Return true if line segments AB and CD intersect
def intersect(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def getPoints(line):
    rho, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))

    return ((x1, y1), (x2, y2))

while True:
    threshold1 = cv.getTrackbarPos("Threshold1", "Parameters")
    threshold2 = cv.getTrackbarPos("Threshold2", "Parameters")

    dilationIterations = cv.getTrackbarPos("DilationIterations", "Parameters")

    kernelSize = cv.getTrackbarPos("KernelSize", "Parameters")

    minArea = cv.getTrackbarPos("Min Area", "Parameters")

    lineThreshold = cv.getTrackbarPos("LineThreshold", "Parameters")

    imgEdges = cv.Canny(imgBlur, threshold1, threshold2)

    kernel = np.ones((kernelSize, kernelSize))
    imgEdgesDilated = cv.dilate(imgEdges, kernel, iterations=dilationIterations)
    contours, hierarchy = cv.findContours(imgEdgesDilated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    imgFinal = img.copy()

    for cnt in contours:
        if cv.contourArea(cnt) >= minArea:
            cv.drawContours(imgFinal, cnt, -1, (255, 0, 255), 2)

    lines = cv.HoughLines(imgEdges, 1, np.pi / 180, lineThreshold)
    if lines is not None:
        print(len(lines))
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            # cv.line(imgFinal, (x1, y1), (x2, y2), (255, 255, 255), 2)

        for i in range(len(lines)-2):
            line1_points = getPoints(lines[i])
            for j in range(i+1, len(lines)-1):
                line2_points = getPoints(lines[j])
                if intersect(*line1_points, *line2_points):
                    for k in range(j+1, len(lines)):
                        line3_points = getPoints(lines[k])
                        if intersect(*line1_points, *line3_points) and intersect(*line2_points, *line3_points):
                            cv.line(imgFinal, line1_points[0], line1_points[1], (255, 255, 255), 2)
                            cv.line(imgFinal, line2_points[0], line2_points[1], (255, 255, 255), 2)
                            cv.line(imgFinal, line3_points[0], line3_points[1], (255, 255, 255), 2)
                            print(line1_points)
                            print(line2_points)
                            print(line3_points)
                            print("----------")



    imgDisplay = np.vstack((np.hstack((img, imgEdges)), np.hstack((imgEdgesDilated, imgFinal))))

    cv.imshow("Edges", imgDisplay)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break





