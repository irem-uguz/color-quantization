import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


def calc_color_distance(arr, center):
    temp = np.subtract(arr, center)
    temp = np.power(temp, 2)
    return np.sqrt(np.sum(temp, axis=1))


def quantize(image_name, k_number, mode):
    im = Image.open(image_name)
    k = k_number
    num_of_iterations = 10
    image_array = np.array(im)
    (w, h, d) = image_array.shape
    image_array = np.reshape(image_array, (w * h, 3))
    color = []

    if mode == 2:
        points = np.random.uniform(0, w*h, k)
        for i in range(k):
            color.append(image_array[int(points[i])])
    elif mode == 1:
        plt.imshow(im)
        points = plt.ginput(k, show_clicks=True)
        for i in range(k):
            color.append(image_array[int(points[i][0]*w+points[i][1])])

    closest_centers = []
    for i in range(num_of_iterations):
        distances_to_centers = []
        clusters = []
        for color_center_index in range(k):
            distances_to_centers.append(calc_color_distance(image_array, color[color_center_index]))
            clusters.append([])
        closest_centers = np.argmin(distances_to_centers, axis=0)
        closest_centers = np.array(closest_centers)
        for index in range(len(image_array)):
            clusters[closest_centers[index]].append(image_array[index])
        for index in range(k):
            if len(clusters[index]) != 0:
                color[index] = np.average(clusters[index], axis=0)

    for index in range(len(image_array)):
        image_array[index] = color[closest_centers[index]]
    arr = np.reshape(image_array, (w, h, 3)).astype('uint8')
    quantized_image = Image.fromarray(arr, 'RGB')
    quantized_image.save(str(k)+"quantized_image"+image_name)


quantize("3.jpg", 32, 1)


def warp(im, H):
    # This part will will calculate the X and Y offsets
    bunchX = [];
    bunchY = []

    tt = np.array([[0], [0], [1]])
    tmp = np.dot(H, tt)
    bunchX.append(tmp[0])
    bunchY.append(tmp[1])

    tt = np.array([[im.shape[0]], [0], [1]])
    tmp = np.dot(H, tt)
    bunchX.append(tmp[0])
    bunchY.append(tmp[1])

    tt = np.array([[0], [im.shape[1]], [1]])
    tmp = np.dot(H, tt)
    bunchX.append(tmp[0])
    bunchY.append(tmp[1])

    tt = np.array([[im.shape[0]], [im.shape[1]], [1]])
    tmp = np.dot(H, tt)
    bunchX.append(tmp[0])
    bunchY.append(tmp[1])

    refX1 = int(np.min(bunchX))
    refY1 = int(np.min(bunchY))
    refX2 = int(np.max(bunchX))
    refY2 = int(np.max(bunchY))

    # Final image whose size is defined by the offsets previously calculated
    final = np.zeros((int(refX2 - refX1), int(refY2 - refY1), 3))
    Hi = np.linalg.inv(H)
    # Iterate over the imagine to forward-transform every pixel
    # for i in range(im.shape[0]):
    #     for j in range(im.shape[1]):
    #         tt = np.array([[i], [j], [1]])
    #         tmp = np.dot(H, tt)
    #         x1 = int(tmp[0] - refX1)
    #         y1 = int(tmp[1] - refY1)
    #         if 0 < y1 < refY2 - refY1 and 0 < x1 < refX2 - refX1:
    #             final[x1][y1] = im[i][j]
    # tmp_final = Image.fromarray(final.astype('uint8'), "RGB")
    # tmp_final.save("_tmp_final.png")
    # Simple Interpolation
    # Interpolate empty pixels from the original image, ignoring pixels outside (extrapolating)
    print("entering for loop with shape "+ str(final.shape))
    for i in range(final.shape[0]):
        for j in range(final.shape[1]):
            # if sum(final[i, j, :]) == 0:
            #     tt = np.array([[i ], [j ], [1]])
            #     tmp = np.dot(Hi, tt)
            #     x1 = int(tmp[0] )
            #     y1 = int(tmp[1] )
            #
            #     if x1 > 0 and y1 > 0 and x1 < im.shape[0] and y1 < im.shape[1]:
            #         final[i, j, :] = im[x1, y1, :]
            # if sum(final[i, j, :]) == 0:
            if sum(final[i, j, :]) == 0:
                tt = np.array([[i + refX1], [j + refY1], [1]])
                tmp = np.dot(Hi, tt)
                tmpX = tmp[0]
                tmpY = tmp[1]
                if im.shape[0] > tmpX > 0 and im.shape[1] > tmpY > 0 :
                    x_ceil = math.ceil(tmpX) if math.ceil(tmpX) < im.shape[0] else im.shape[0]-1
                    y_ceil = math.ceil(tmpY) if math.ceil(tmpY) < im.shape[1] else im.shape[1]-1
                    pointsX = [math.floor(tmpX), math.floor(tmpX), x_ceil, x_ceil]
                    pointsY = [math.floor(tmpY), y_ceil, y_ceil, math.floor(tmpY)]
                    red = np.uint8(round((int(im[pointsX[0]][pointsY[0]][0]) + int(im[pointsX[1]][pointsY[1]][0]) +
                                         int(im[pointsX[2]][pointsY[2]][0]) + int(im[pointsX[3]][pointsY[3]][0])) / 4.0))
                    green = np.uint8(round((int(im[pointsX[0]][pointsY[0]][1]) + int(im[pointsX[1]][pointsY[1]][1]) +
                                         int(im[pointsX[2]][pointsY[2]][1]) + int(im[pointsX[3]][pointsY[3]][1])) / 4.0))
                    blue = np.uint8(round((int(im[pointsX[0]][pointsY[0]][2]) + int(im[pointsX[1]][pointsY[1]][2]) +
                                         int(im[pointsX[2]][pointsY[2]][2]) + int(im[pointsX[3]][pointsY[3]][2])) / 4.0))
                    pixel = np.array([red, green, blue])
                    final[i][j] = pixel
                    # #print(str(calcValue(pointsX[0], pointsY[0], im)) + " and " + str(im[pointsX[0]][pointsY[0]]) + " and " + str(invCalcValue(calcValue(pointsX[0], pointsY[0], im))))
                    # values = [calcValue(pointsX[0], pointsY[0], im), calcValue(pointsX[1], pointsY[1], im),
                    #           calcValue(pointsX[2], pointsY[2], im), calcValue(pointsX[3], pointsY[3], im)]
                    # pixel_func = interpolate.interp2d(pointsX, pointsY, values, kind='linear')
                    # pixel = invCalcValue(pixel_func(tmpX, tmpY))
                    # final[i][j] = pixel
                    # final[i][j] = im[tmpX][tmpY]
            #print(str(invCalcValue(pixel_func(tmpX, tmpY)[0]))+ " and "+ str(im[int(tmpX)][int(tmpY)]))
    print("exit for loop")
    fin_im = Image.fromarray(final.astype('uint8'), "RGB")
    fin_im.save("final1.png")
    return (fin_im, final)



Öbür Kod
def warp(im, H):
    # This part will will calculate the X and Y offsets
    bunchX = [];
    bunchY = []

    tt = np.array([[1], [1], [1]])
    tmp = np.dot(H, tt)
    bunchX.append(tmp[0] / tmp[2])
    bunchY.append(tmp[1] / tmp[2])

    tt = np.array([[im.shape[1]], [1], [1]])
    tmp = np.dot(H, tt)
    bunchX.append(tmp[0] / tmp[2])
    bunchY.append(tmp[1] / tmp[2])

    tt = np.array([[1], [im.shape[0]], [1]])
    tmp = np.dot(H, tt)
    bunchX.append(tmp[0] / tmp[2])
    bunchY.append(tmp[1] / tmp[2])

    tt = np.array([[im.shape[1]], [im.shape[0]], [1]])
    tmp = np.dot(H, tt)
    bunchX.append(tmp[0] / tmp[2])
    bunchY.append(tmp[1] / tmp[2])

    refX1 = int(np.min(bunchX))
    refY1 = int(np.min(bunchY))
    refX2 = int(np.max(bunchX))
    refY2 = int(np.max(bunchY))
    print(str(refX1) + " and "+ str(refX2) + " and " + str(refY1) + " and "+ str(refY2))

    # Final image whose size is defined by the offsets previously calculated
    final = np.zeros((int(refY2 - refY1), int(refX2 - refX1), 3))
    Hi = np.linalg.inv(H)
    # Iterate over the imagine to forward-transform every pixel
    # for i in range(im.shape[0]):
    #     for j in range(im.shape[1]):
    #         tt = np.array([[i], [j], [1]])
    #         tmp = np.dot(H, tt)
    #         x1 = int(tmp[0] - refX1)
    #         y1 = int(tmp[1] - refY1)
    #         if 0 < y1 < refY2 - refY1 and 0 < x1 < refX2 - refX1:
    #             final[x1][y1] = im[i][j]
    # tmp_final = Image.fromarray(final.astype('uint8'), "RGB")
    # tmp_final.save("_tmp_final.png")
    # Simple Interpolation
    # Interpolate empty pixels from the original image, ignoring pixels outside (extrapolating)
    print("entering for loop with shape "+ str(final.shape))
    for i in range(final.shape[0]):
        for j in range(final.shape[1]):
            # if sum(final[i, j, :]) == 0:
            #     tt = np.array([[i ], [j ], [1]])
            #     tmp = np.dot(Hi, tt)
            #     x1 = int(tmp[0] )
            #     y1 = int(tmp[1] )
            #
            #     if x1 > 0 and y1 > 0 and x1 < im.shape[0] and y1 < im.shape[1]:
            #         final[i, j, :] = im[x1, y1, :]
            # if sum(final[i, j, :]) == 0:
            if sum(final[i, j, :]) == 0:
                tt = np.array([[j + refX1], [i + refY1], [1]])
                tmp = np.dot(Hi, tt)
                tmpX = tmp[0] / tmp[2]
                tmpY = tmp[1] / tmp[2]
                if im.shape[1] > tmpX > 0 and im.shape[0] > tmpY > 0 :
                    x_ceil = math.ceil(tmpX) if math.ceil(tmpX) < im.shape[1] else im.shape[1]-1
                    y_ceil = math.ceil(tmpY) if math.ceil(tmpY) < im.shape[0] else im.shape[0]-1
                    pointsX = [math.floor(tmpX), math.floor(tmpX), x_ceil, x_ceil]
                    pointsY = [math.floor(tmpY), y_ceil, y_ceil, math.floor(tmpY)]
                    red = np.uint8(round((int(im[pointsY[0]][pointsX[0]][0]) + int(im[pointsY[1]][pointsX[1]][0]) +
                                         int(im[pointsY[2]][pointsX[2]][0]) + int(im[pointsY[3]][pointsX[3]][0])) / 4.0))
                    green = np.uint8(round((int(im[pointsY[0]][pointsX[0]][1]) + int(im[pointsY[1]][pointsX[1]][1]) +
                                         int(im[pointsY[2]][pointsX[2]][1]) + int(im[pointsY[3]][pointsX[3]][1])) / 4.0))
                    blue = np.uint8(round((int(im[pointsY[0]][pointsX[0]][2]) + int(im[pointsY[1]][pointsX[1]][2]) +
                                         int(im[pointsY[2]][pointsX[2]][2]) + int(im[pointsY[3]][pointsX[3]][2])) / 4.0))
                    pixel = np.array([red, green, blue])
                    final[i][j] = pixel
                    # #print(str(calcValue(pointsX[0], pointsY[0], im)) + " and " + str(im[pointsX[0]][pointsY[0]]) + " and " + str(invCalcValue(calcValue(pointsX[0], pointsY[0], im))))
                    # values = [calcValue(pointsX[0], pointsY[0], im), calcValue(pointsX[1], pointsY[1], im),
                    #           calcValue(pointsX[2], pointsY[2], im), calcValue(pointsX[3], pointsY[3], im)]
                    # pixel_func = interpolate.interp2d(pointsX, pointsY, values, kind='linear')
                    # pixel = invCalcValue(pixel_func(tmpX, tmpY))
                    # final[i][j] = pixel
                    # final[i][j] = im[tmpX][tmpY]
            #print(str(invCalcValue(pixel_func(tmpX, tmpY)[0]))+ " and "+ str(im[int(tmpX)][int(tmpY)]))
    print("exit for loop")
    fin_im = Image.fromarray(final.astype('uint8'), "RGB")
    fin_im.save("final1.png")
    return (fin_im, final)