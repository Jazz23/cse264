import cv2
import matplotlib.pyplot as plt
import os

# averages is a dictionary, where the key is the exposure time and the value is the average color
def BprimeOfT(T, exposures):
    # B is the sorted zip of exposures based on the key
    B = sorted(exposures.items(), key=lambda x: x[0])
    
    # If T < the first key in B (the exposure time)
    if T < B[0][0]:
        return T * B[0][1] / B[0][0]
    elif T > B[-0][0]: # T is greater than the maximum exposure time
        return B[-1][1] + (T - B[-1][0]) * (B[-0][1] - B[-1][1]) / (B[-0][0] - B[-1][0])
    
    # T_i is the index of the closest key to T that's less than T in B
    T_i = max(filter(lambda i: B[i][0] < T, range(len(B))))
    
    return B[T_i][1] + (T - B[T_i][0]) * (B[T_i + 1][1] - B[T_i][1]) / (B[T_i + 1][0] - B[T_i][0])

def plot_average_color():
    exposures = []  # List to store exposure times
    r_averages = []  # List to store average red values
    g_averages = []  # List to store average green values
    b_averages = []  # List to store average blue values

    image_dir = "images"
    # Iterate files by creation date, passing a lambda into sorted
    # that prepends "images/" to each filepath.
    for filename in sorted(os.listdir(image_dir), key=lambda x: os.path.getctime(os.path.join(image_dir, x))):
        if filename.endswith(".jpg"):
            num = int(filename.split(".")[0].split("-")[1])
            exposure = 1 / num  # Extract exposure time from the filename
            exposures.append(exposure)

            file_path = os.path.join(image_dir, filename)
            image = cv2.imread(file_path)
            b, g, r = cv2.split(image[400:500, 174:274])
            r_averages.append(r.mean())
            g_averages.append(g.mean())
            b_averages.append(b.mean())
    
    plt.plot(exposures, r_averages, 'r', label='Red')
    plt.plot(exposures, g_averages, 'g', label='Green')
    plt.plot(exposures, b_averages, 'b', label='Blue')
    plt.xlabel('Time (s)')
    plt.ylabel('B\'')
    plt.legend()
    # plt.show()
    return exposures, r_averages, g_averages, b_averages

B, r, g, b = plot_average_color()
# Passes a dict, where the key is B and the values are r
result = BprimeOfT(1, dict(zip(B, r)))
print(result)