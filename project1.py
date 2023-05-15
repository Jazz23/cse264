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

# Coords is the coords of white square
def greyCardMatrix(img, coords):
    # Get the average rgb values at the patch coords
    b, g, r = cv2.split(img[coords[1]:coords[1] + 40, coords[0]:coords[0] + 40])
    # Return a matrix such that D * [r, g, b] = [180, 180, 180]
    return [200 / b.mean(), 200 / g.mean(), 200 / r.mean()]

def applyGreyCard(img, D):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i][j][0] *= D[0]
            img[i][j][1] *= D[1]
            img[i][j][2] *= D[2]
    return img

# Input is the center point of the white, blue, yellow, and red patches if img,
# linearized.
# Returns the average linearized rgb values of each of the 4 patches
def getCValues(img, exposures, coords: tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]],
               greyCard: bool = False, greyWorldCoords = None):
    result = []
    for coord in coords:
        patch = img[coord[1]:coord[1]+20, coord[0]:coord[0]+20]
        linearized = linearize(patch, exposures)
        if greyCard:
            D = greyCardMatrix(img, coords[0]) # D is [blue, green, red] multipliers
            linearized = applyGreyCard(linearized, D) # First coord is white
        elif greyWorldCoords != None:
            D = getGreyWorldD(img, greyWorldCoords)
            linearized = applyGreyCard(linearized, D)
            
        bl, gl, rl = cv2.split(linearized)
        # append a triple representing the mean r g b of linearized
        result.append((bl.mean(), gl.mean(), rl.mean()))
    return result

# Thank you chat gpt!
# For each subarray, the first four coordinate pairs are the white, ... squares
# The next 2 pairs are the top left and bottom right of the square box as a whole
def parseCoords(folder):
    coordinates = []
    squareCoords = []
    subarray = []

    with open(os.path.join(folder, "coords.txt"), 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip()
            if line:
                x, y = map(int, line.split(','))
                subarray.append((x, y))
                if len(subarray) == 6:
                    coordinates.append(subarray[:4])
                    squareCoords.append(subarray[-2:])
                    subarray = []
                    
    return coordinates, squareCoords

def getGreyWorldD(img, coords):
    # Get the average rgb values at the patch coords
    b, g, r = cv2.split(img[coords[0][0]:coords[1][0], coords[0][1]:coords[1][1]])
    # Return a matrix such that D * [r, g, b] = [180, 180, 180]
    return [200 / b.mean(), 200 / g.mean(), 200 / r.mean()]
    

# coords is [(), ()] the top left, bottom right coords
def applyGreyWorld(img, coords):
    D = getGreyWorldD(img, coords)
    

# Returns an array of length 4, with each entry being (R', G') for that square
# Array1 (4): Each color
# Array2 (5): Each image
# Array3 (2): R', G'
# [ [(R', G'), (R', G'), (), (), ()], , , ]
def gatherCFromFolder(folder, exposures, greyCard = False, greyWorld = False):
    result = [[] for _ in range(4)]
    coords, sqCoords = parseCoords(folder) # Square positions for w, ... and coordinates for square
    
    for i in range(5):
        # For each image/angle
        img = cv2.imread(os.path.join(folder, f"{i}.jpg"))
        # cs is an array of lenght 4 of triples, b g r linearized average of each coord
        greyWorldCoords = sqCoords[i] if greyWorld else None
        cs = getCValues(img, exposures, coords[i], greyCard, greyWorldCoords)
        totalWhite = cs[0][0] + cs[0][1] + cs[0][2]
        rgWhite = (cs[0][2] / totalWhite, cs[0][1] / totalWhite)
        
        totalCyan = cs[0][0] + cs[0][1] + cs[0][2]
        rgCyan = (cs[0][2] / totalCyan, cs[0][1] / totalCyan)
        
        totalYellow = cs[0][0] + cs[0][1] + cs[0][2]
        rgYellow = (cs[0][2] / totalYellow, cs[0][1] / totalYellow)
        
        totalMagenta = cs[0][0] + cs[0][1] + cs[0][2]
        rgMagenta = (cs[0][2] / totalMagenta, cs[0][1] / totalMagenta)
        
        result[0].append(rgWhite) # Append this pair to white's list of 5 rgs
        result[1].append(rgCyan)
        result[2].append(rgYellow)
        result[3].append(rgMagenta)
    return result
    

# def plot(cs, ilSymbol, ):
    
    
B, b, g, r = get_averages()
exposures = [dict(zip(B, b)), dict(zip(B, g)), dict(zip(B, r))]
test = gatherCFromFolder("images/Part2/wb0il1", exposures, greyWorld = True)
print(test)