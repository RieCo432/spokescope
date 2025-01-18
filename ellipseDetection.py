import cv2
import numpy as np


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
cv2.createTrackbar("Threshold1", "Parameters", 145, 1000, lambda _: None)
cv2.createTrackbar("Threshold2", "Parameters", 31, 1000, lambda _: None)
cv2.createTrackbar("Gradient Threshold", "Parameters", 0, 1000, lambda _: None)
cv2.createTrackbar("Area Threshold", "Parameters", 6500, 50000, lambda _: None)
cv2.createTrackbar("DilationIterations", "Parameters", 2, 10, lambda _: None)
cv2.createTrackbar("Dil Kernel", "Parameters", 3, 15, lambda _: None)
cv2.createTrackbar("Sobel Kernel", "Parameters", 12, 15, lambda _: None)




img = cv2.imread("images/input/img1.jpg")

imgSize = img.shape
maxHeight = 500
maxWidth = 900

widthFactor = maxWidth / imgSize[1]
heightFactor = maxHeight / imgSize[0]

factor = min(widthFactor, heightFactor)
img = cv2.resize(img, (0,0), fx=factor, fy=factor)

imgNorm = img / 255.0
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

while True:

    threshold1 = cv2.getTrackbarPos("Threshold1", "Parameters")
    threshold2 = cv2.getTrackbarPos("Threshold2", "Parameters")
    gradientThreshold = cv2.getTrackbarPos("Gradient Threshold", "Parameters")
    areaThreshold = cv2.getTrackbarPos("Area Threshold", "Parameters")
    dilationIterations = cv2.getTrackbarPos("DilationIterations", "Parameters")
    dilationKernelSize = cv2.getTrackbarPos("Dil Kernel", "Parameters")
    sobelKernelSize = cv2.getTrackbarPos("Sobel Kernel", "Parameters")

    dilationKernelSize = max(dilationKernelSize, 1)

    kernel = np.ones((dilationKernelSize, dilationKernelSize))

    # imgBlur = cv2.GaussianBlur(imgGray, (2*kernelSize+1, 2*kernelSize+1), 1)

    imgEdgesRaw = cv2.Canny(img, threshold1, threshold2)

    imgEdges = cv2.dilate(imgEdgesRaw, kernel, iterations=dilationIterations)


    grad_x = cv2.Sobel(imgEdges, cv2.CV_64F, 1, 0, ksize=2 * sobelKernelSize + 1)
    grad_y = cv2.Sobel(imgEdges, cv2.CV_64F, 0, 1, ksize=2 * sobelKernelSize + 1)
    grad = np.sqrt(grad_x ** 2 + grad_y ** 2)
    imgSobelEdges = (grad * 255 / grad.max()).astype(np.uint8)

    negative_x = grad_x < -gradientThreshold
    positive_x = grad_x > gradientThreshold
    negative_y = grad_y < -gradientThreshold
    positive_y = grad_y > gradientThreshold

    class1 = negative_x & negative_y
    class2 = positive_x & positive_y
    class3 = negative_x & positive_y
    class4 = positive_x & negative_y

    class1PointsImg = np.array(np.where(class1, 1.0, 0), dtype=np.uint8)
    class2PointsImg = np.array(np.where(class2, 1.0, 0), dtype=np.uint8)
    class3PointsImg = np.array(np.where(class3, 1.0, 0), dtype=np.uint8)
    class4PointsImg = np.array(np.where(class4, 1.0, 0), dtype=np.uint8)



    imgClass1 = np.where(class1[:, :, np.newaxis], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    imgClass2 = np.where(class2[:, :, np.newaxis], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0])
    imgClass3 = np.where(class3[:, :, np.newaxis], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0])
    imgClass4 = np.where(class4[:, :, np.newaxis], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0])

    imgClassesA = imgClass1 + imgClass2
    imgClassesB = imgClass3 + imgClass4

    imgComponents = imgNorm.copy()

    (numLabels1, labels1, stats1, centroids1) = cv2.connectedComponentsWithStats(class1PointsImg, connectivity=8)
    arcs1 = []
    for i in range(numLabels1):
        if stats1[i,2]*stats1[i,3] > areaThreshold:
            cv2.rectangle(imgComponents, stats1[i, 0:2], stats1[i, 0:2]+stats1[i, 2:4], [1.0,0,0], 1)
            arcs1.append(np.transpose(np.asarray(labels1==i).nonzero()))

    (numLabels2, labels2, stats2, centroids2) = cv2.connectedComponentsWithStats(class2PointsImg, connectivity=8)
    arcs2 = []
    for i in range(numLabels2):
        if stats2[i,2]*stats2[i,3] > areaThreshold:
            cv2.rectangle(imgComponents, stats2[i, 0:2], stats2[i, 0:2] + stats2[i, 2:4], [0,1.0,0], 1)
            arcs2.append(np.transpose(np.asarray(labels2==i).nonzero()))

    (numLabels3, labels3, stats3, centroids3) = cv2.connectedComponentsWithStats(class3PointsImg, connectivity=8)
    arcs3 = []
    for i in range(numLabels3):
        if stats3[i,2]*stats3[i,3] > areaThreshold:
            cv2.rectangle(imgComponents, stats3[i, 0:2], stats3[i, 0:2] + stats3[i, 2:4], [0,0,1.0], 1)
            arcs3.append(np.transpose(np.asarray(labels3==i).nonzero()))

    (numLabels4, labels4, stats4, centroids4) = cv2.connectedComponentsWithStats(class4PointsImg, connectivity=8)
    arcs4 = []
    for i in range(numLabels4):
        if stats4[i,2]*stats4[i,3] > areaThreshold:
            cv2.rectangle(imgComponents, stats4[i, 0:2], stats4[i, 0:2] + stats4[i, 2:4], [1.0,1.0,0], 1)
            arcs4.append(np.transpose(np.asarray(labels4==i).nonzero()))

    # allGradClasses = np.where(class1 | class2 | class3 | class4, 1.0, 0)

    imgDisplay = img_quadro(imgEdges, imgClassesA, imgClassesB, imgComponents)
    cv2.imshow("Arcs", imgDisplay)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

