import time

import cv2
import numpy as np
from time import sleep


class Arc:

    points = []
    direction = 0
    convexity = 0
    area = 0
    center_of_mass = np.array([])
    center = np.array([])
    isolated = None
    cutout = None

    def __init__(self, direction, isolated):
        self.direction = direction
        mean = np.zeros_like(isolated)

        for x in range(isolated.shape[0]):
            try:
                avg_y = np.mean(np.transpose(np.asarray(arcIsolated[x-1:x+2,:]).nonzero())[:,1])
                if avg_y > 0:
                    mean[x-1:x+2, int(avg_y)-1:int(avg_y)+2] = np.ones((3,3))

            except:
                pass


        for y in range(isolated.shape[1]):
            try:
                avg_x = np.mean(np.transpose(np.asarray(arcIsolated[:,y-1:y+2]).nonzero())[:,0])
                if avg_x > 0:
                    mean[int(avg_x)-1:int(avg_x)+2, y-1:y+2] = np.ones((3,3))
            except:
                pass

        self.isolated = isolated
        self.cutout = mean * isolated

        points = np.transpose(np.asarray(self.cutout).nonzero())
        self.points = sorted(points, key=lambda p: p[0])
        points_sorted_y = sorted(points, key=lambda p: p[1])



        points_x = [point[0] for point in points]
        points_y = [point[1] for point in points]

        avg_x = int(round(sum(points_x) / len(points_x)))
        avg_y = int(round(sum(points_y) / len(points_y)))

        center_x = int(round(self.points[0][0] + self.points[-1][0]) / 2)
        center_y = int(round(self.points[0][1] + self.points[-1][1]) / 2)

        self.center_of_mass = np.array([avg_y, avg_x])
        self.center = np.array([center_y, center_x])

        self.area = (self.points[-1][0] - self.points[0][0]) * (points_sorted_y[-1][1] - points_sorted_y[0][1])

        self.set_convexity()

    def set_convexity(self):
        a = self.points
        n = len(self.points)
        left = self.points[0]
        right = self.points[-1]
        current_x = left[0]
        area_o = 0
        for i in range(0, n):
            if a[i][0] != current_x:
                area_o += abs(a[i][1] - left[1])
                current_x = a[i][0]
        area_u = self.area - n - area_o
        if area_u > area_o*1.1:
            self.convexity = 1
        elif area_u < area_o*0.9:
            self.convexity = -1
        else:
            self.convexity = None



    def quadrant(self):
        if self.convexity == 1 and self.direction == 1:
            return 1
        elif self.convexity == -1 and self.direction == 1:
            return 2
        elif self.convexity == 1 and self.direction == -1:
            return 3
        elif self.convexity == -1 and self.direction == -1:
            return 4


def img_quadro(img1, img2 = None, img3 = None, img4 = None):

    if img1 is not None and len(img1.shape) == 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)

    if img2 is not None and len(img2.shape) == 2:
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    if img3 is not None and len(img3.shape) == 2:
        img3 = cv2.cvtColor(img3, cv2.COLOR_GRAY2BGR)

    if img4 is not None and len(img4.shape) == 2:
        img4 = cv2.cvtColor(img4, cv2.COLOR_GRAY2BGR)

    if img2 is None and img3 is None and img4 is None:
        return img1

    if img3 is None and img4 is None:
        return np.hstack((img1, img2))

    if img4 is None:
        img4 = img3.copy()

    return np.vstack((np.hstack((img1, img2)), np.hstack((img3, img4))))


cv2.namedWindow("Parameters")
cv2.resizeWindow("Parameters", 800, 600)
cv2.createTrackbar("Threshold1", "Parameters", 235, 1000, lambda _: None)
cv2.createTrackbar("Threshold2", "Parameters", 225, 1000, lambda _: None)
cv2.createTrackbar("Gradient Threshold", "Parameters", 0, 1000, lambda _: None)
cv2.createTrackbar("Area Threshold", "Parameters", 380, 50000, lambda _: None)
cv2.createTrackbar("DilationIterations", "Parameters", 1, 10, lambda _: None)
cv2.createTrackbar("Dil Kernel", "Parameters", 3, 15, lambda _: None)
cv2.createTrackbar("Sobel Kernel", "Parameters", 2, 15, lambda _: None)




img = cv2.imread("images/input/img2.jpg")

imgSize = img.shape
maxHeight = 500
maxWidth = 900

widthFactor = maxWidth / imgSize[1]
heightFactor = maxHeight / imgSize[0]

factor = min(widthFactor, heightFactor)
img = cv2.resize(img, (0,0), fx=factor, fy=factor)

img_area = img.shape[0] * img.shape[1]

imgNorm = img / 255.0
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
arcNum = -1

while True:

    threshold1 = cv2.getTrackbarPos("Threshold1", "Parameters")
    threshold2 = cv2.getTrackbarPos("Threshold2", "Parameters")
    gradientThreshold = cv2.getTrackbarPos("Gradient Threshold", "Parameters")
    areaThreshold = max(cv2.getTrackbarPos("Area Threshold", "Parameters"), 10)
    dilationIterations = cv2.getTrackbarPos("DilationIterations", "Parameters")
    dilationKernelSize = cv2.getTrackbarPos("Dil Kernel", "Parameters")
    sobelKernelSize = cv2.getTrackbarPos("Sobel Kernel", "Parameters")

    dilationKernelSize = max(dilationKernelSize, 1)

    kernel = np.ones((dilationKernelSize, dilationKernelSize))

    # imgBlur = cv2.GaussianBlur(imgGray, (2*kernelSize+1, 2*kernelSize+1), 1)

    imgEdgesRaw = cv2.Canny(img, threshold1, threshold2)

    imgEdges = cv2.dilate(imgEdgesRaw, kernel, iterations=dilationIterations)


    grad_x = cv2.Sobel(imgEdges, cv2.CV_32F, 1, 0, ksize=2 * sobelKernelSize + 1)
    grad_y = cv2.Sobel(imgEdges, cv2.CV_32F, 0, 1, ksize=2 * sobelKernelSize + 1)
    grad = np.sqrt(grad_x ** 2 + grad_y ** 2)
    imgSobelEdges = (grad * 255 / grad.max()).astype(np.uint8)

    negative_x = grad_x < -gradientThreshold
    positive_x = grad_x > gradientThreshold
    negative_y = grad_y < -gradientThreshold
    positive_y = grad_y > gradientThreshold

    class1 = (negative_x & negative_y) | (positive_x & positive_y)
    class2 = (negative_x & positive_y) | (positive_x & negative_y)

    class1PointsImg = np.array(np.where(class1, 255, 0), dtype=np.uint8)
    class2PointsImg = np.array(np.where(class2, 255, 0), dtype=np.uint8)

    imgClass1 = np.where(class1[:, :, np.newaxis], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    imgClass2 = np.where(class2[:, :, np.newaxis], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0])

    imgAllClasses = imgClass1 + imgClass2

    imgComponents = imgNorm.copy()

    all_arcs = []

    (numLabels1, labels1, stats1, centroids1) = cv2.connectedComponentsWithStats(class1PointsImg, connectivity=8)
    for i in range(numLabels1):
        area = stats1[i,2]*stats1[i,3]
        if area > areaThreshold and area < img_area / 2:
            arcIsolated = np.where(labels1 == i, 1.0, 0.0)
            arc = Arc(-1, arcIsolated)
            if arc.convexity is not None:
                cv2.rectangle(imgComponents, stats1[i, 0:2], stats1[i, 0:2] + stats1[i, 2:4], [1.0, 0, 0], 1)
                all_arcs.append(arc)

    (numLabels2, labels2, stats2, centroids2) = cv2.connectedComponentsWithStats(class2PointsImg, connectivity=8)
    for i in range(numLabels2):
        area = stats2[i,2]*stats2[i,3]
        if area > areaThreshold and area < img_area / 2:
            arcIsolated = np.where(labels2 == i, 1.0, 0.0)
            arc = Arc(1, arcIsolated)
            if arc.convexity is not None:
                cv2.rectangle(imgComponents, stats2[i, 0:2], stats2[i, 0:2] + stats2[i, 2:4], [0, 1.0, 0], 1)
                all_arcs.append(arc)



    for arc in all_arcs:
        color = [1.0,1.0,1.0]
        if arc.quadrant() == 1:
            color = [1.0, 0.0, 0.0]
        elif arc.quadrant() == 2:
            color = [0.0, 1.0, 0.0]
        elif arc.quadrant() == 3:
            color = [0.0, 0.0, 1.0]
        elif arc.quadrant() == 4:
            color = [1.0, 1.0, 0.0]
        # for point in arc.points:
        #    imgComponents[point[0], point[1], :] = color

        cv2.circle(imgComponents, arc.center_of_mass, 2, [0.0, 1.0, 1.0], 2)
        cv2.circle(imgComponents, arc.center, 2, [1.0, 0.0, 1.0], 2)

        # imgComponents[arc.isolated == 1.0,:] = [1.0, 1.0, 1.0]
        imgComponents[arc.cutout == 1.0, :] = color


    imgDisplay = img_quadro(imgEdges, imgSobelEdges, imgAllClasses, imgComponents)
    cv2.imshow("Arcs", imgDisplay)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

