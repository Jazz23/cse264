import cv2
import matplotlib.pyplot as plt
import os

# Returns brightness for a given epoxure time T
# averages is a dictionary, where the key is the exposure time and the value is the average color
def BprimeOfT(T, exposures):
    # B is a dict: key is Time, value is Brightness
    B = sorted(exposures.items(), key=lambda x: x[0]) # Sort by Time
    
    # If T < the first key in B (the exposure time)
    if T < B[0][0]:
        return T * B[0][1] / B[0][0]
    elif T > B[-0][0]: # T is greater than the maximum exposure time
        return B[-1][1] + (T - B[-1][0]) * (B[-0][1] - B[-1][1]) / (B[-0][0] - B[-1][0])
    
    # T_i is the index of the closest key to T that's less than T in B
    T_i = max(filter(lambda i: B[i][0] < T, range(len(B))))
    
    return B[T_i][1] + (T - B[T_i][0]) * (B[T_i + 1][1] - B[T_i][1]) / (B[T_i + 1][0] - B[T_i][0])

# Returns exposure time for given Brightness
def gOf(b: float, exposures: dict[float, float]):
    # Key is Time, value is Brightness
    B = sorted(exposures.items(), key=lambda x: x[1]) # Sort by brightness
    
    # If B' < The smallest brightness in B
    if b < B[0][1]:
        return b * B[0][0] / B[0][1]
    elif b > B[-0][1]: # T is greater than the maximum brightnes
        return B[-1][0] + (b - B[-1][1]) * (B[-0][0] - B[-1][0]) / (B[-0][1] - B[-1][1])
    
    T_i = max(filter(lambda i: B[i][1] < b, range(len(B))))
    
    return B[T_i][0] + (b - B[T_i][1]) * (B[T_i+1][0]-B[T_i][0]) / (B[T_i+1][1] - B[T_i][1])
    
    
def linearize(img, exposures: list[dict[float, float]]):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            alpha1 = 255 / gOf(255, exposures[0])
            alpha2 = 255 / gOf(255, exposures[1])
            alpha3 = 255 / gOf(255, exposures[2])
            img[i][j] = (alpha1 * gOf(img[i][j][0], exposures[0]),
                         alpha2 * gOf(img[i][j][1], exposures[1]),
                         alpha3 * gOf(img[i][j][2], exposures[2]))
    return img
    

def get_averages():
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
    
    # plt.plot(exposures, r_averages, 'r', label='Red')
    # plt.plot(exposures, g_averages, 'g', label='Green')
    # plt.plot(exposures, b_averages, 'b', label='Blue')
    # plt.xlabel('Time (s)')
    # plt.ylabel('B\'')
    # plt.legend()
    # plt.show()
    return exposures, b_averages, g_averages, r_averages


# Input is the center point of the white, blue, yellow, and red patches if img
# Returns the average linearized rgb values of each of the 4 patches
def getCValues(img, coords: tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]]):
    result = []
    B, b, g, r = get_averages()
    exposures = [dict(zip(B, b)), dict(zip(B, g)), dict(zip(B, r))]
    for coord in coords:
        patch = img[coord[1]:coord[1]+20, coord[0]:coord[0]+20]
        linearized = linearize(patch, exposures)
        bl, gl, rl = cv2.split(linearized)
        # append a triple representing the mean r g b of linearized
        result.append((bl.mean(), gl.mean(), rl.mean()))
    return result

def greyCardMatrix(img, coords):
    # Get the average rgb values at the patch coords
    b, g, r = cv2.split(img[coords[1]:coords[1] + 40, coords[0]:coords[0] + 40])
    # Return a matrix such that D * [r, g, b] = [180, 180, 180]
    return [200 / b.mean(), 200 / g.mean(), 200 / r.mean()]

# Passes a dict, where the key is B and the values are r
il1img, coords1 = cv2.imread("images/Part2/wb0il1/0.jpg"), (388, 1044)
il2img, coords2 = cv2.imread("images/Part2/wb0il2/0.jpg"), (388, 972)
il3img, coords3 = cv2.imread("images/Part2/wb0il3/0.jpg"), (324, 996)

il1D = greyCardMatrix(il1img, coords1)
il2D = greyCardMatrix(il2img, coords2)
il3D = greyCardMatrix(il3img, coords3)



# Write the image to test.jpg
# cv2.imwrite("test.jpg", img) # 680, 1088
# cv2.imwrite("test3.jpg", linearize(img[680:700, 1088:1108], [dict(zip(B, b)), dict(zip(B, g)), dict(zip(B, r))]))
# write the patch from 684, 1104 to 704, 1124

# get the patch from 700, 1088 to 720, 1108
cs = getCValues(img, ((684, 1104), (684, 1084), (704, 1084), (704, 1104)))
# Plot (green, red) for each triple in cs, with each point as a red X
plt.plot([x[1] for x in cs], [x[2] for x in cs], 'x', color='red')
plt.xlabel('Green')
plt.ylabel('Red')
plt.show()
cv2.waitKey(0)
# Display the patch