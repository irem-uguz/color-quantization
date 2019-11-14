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