import argparse
import cv2
import numpy as np
import os
import math
import matplotlib.pyplot as plt
import sys
import shutil

directory = "./db"

def recalcDescriptors():
    harris_descriptors ={}
    histograms = {}
    # Iterate over all the images in the directory
    for filename in os.listdir(directory):
        # Check if the file is a .jpg file
        if filename.endswith(".jpg"):
            # Full path to the image file
            filepath = os.path.join(directory, filename)

            # Load the image
            img = cv2.imread(filepath)

            # Split the image into its color channels
            channels = cv2.split(img)

            # Colors to iterate over
            colors = ("b", "g", "r")

            # Convert the image to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Calculate the Harris descriptor
            harris = cv2.cornerHarris(gray, 2, 3, 0.04)

            # Store the Harris descriptor in the dictionary
            harris_descriptors[filename] = harris
            # Iterate over the colors and plot the histograms
            hist_dict = {}
            for i, color in enumerate(colors):
                # Calculate the channel index
                channel_index = colors.index(color)

                # Calculate the histogram for the selected color channel
                hist = cv2.calcHist([img], [channel_index], None, [256], [0, 256])
                hist /= hist.sum()
                hist_dict[color] = hist

            histograms[filename] = hist_dict
    combined = {}
    for filename in os.listdir(directory):
    # Check if the file is a .jpg file
      if filename.endswith(".jpg"):
        combined[filename] = {'histogram':histograms[filename],'harris':harris_descriptors[filename]}
    # Save combined descriptors
    np.save('descriptors.npy', combined)
    return



def addToDb(file_path,db):
    print(f"Adding {file_path} to the database...")
    colors = ("b", "g", "r")
    # Check if the destination directory exists, if not, create it
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Extract the filename from the file path
    filename = os.path.basename(file_path)
    new_filename = ""
    # Construct the destination path
    dest_path = os.path.join(directory, filename)

    # Check if the file already exists in the destination directory
    if os.path.exists(dest_path):
        # If the file exists, generate a new name
        base, ext = os.path.splitext(filename)
        i = 1
        while True:
            new_filename = f"{base}_{i}{ext}"
            dest_path = os.path.join(directory, new_filename)
            if not os.path.exists(dest_path):
                break
            i += 1

    # Copy the file to the destination directory
    shutil.copy(file_path, dest_path)
    aimg = cv2.imread(dest_path)
    hist_dict = {}
    for i, color in enumerate(colors):
    # Calculate the channel index
      channel_index = colors.index(color)
      # Calculate the histogram for the selected color channel
      hist = cv2.calcHist([aimg], [channel_index], None, [256], [0, 256])
      hist_dict[color] = hist
    imghist = hist_dict
    gray = cv2.cvtColor(aimg, cv2.COLOR_BGR2GRAY)
  # Calculate the Harris descriptor
    imgharr = cv2.cornerHarris(gray, 2, 3, 0.04)    
    db[new_filename] = {"histogram":imghist, "harris":imgharr}
    np.save('descriptors.npy', db)




def normalize_array(array):
    min_val = np.min(array)
    max_val = np.max(array)
    normalized_array = (array - min_val) / (max_val - min_val)
    return normalized_array


def calculate_hist_d(h1,h2):
  colors = ("b", "g", "r")
  distance = 0
  for color in colors:
    distance += cv2.compareHist(h1[color], h2[color], cv2.HISTCMP_CHISQR)
  return distance

def calculate_harris_d(h1,h2):
  min_rows = min(h1.shape[0], h2.shape[0])
  min_cols = min(h1.shape[1], h2.shape[1])
  harris1 = h1[:min_rows, :min_cols]
  harris2 = h2[:min_rows, :min_cols]
  distance = np.sum((harris1 - harris2)**2)
  return distance

def calculate_vector_distance_to_origin(v):
    x, y = v
    distance = math.sqrt(x**2 + y**2)
    return distance


def find_min_values(files,histV,harrV):
    min_values = []  # List to store tuples of (index, value)
    for i in range(len(files)):
        dV = (histV[i], harrV[i])
        res = calculate_vector_distance_to_origin(dV)
        if len(min_values) < 5:
            # If we haven't found 5 min values yet, simply append
            min_values.append((i, res))
            min_values.sort(key=lambda x: x[1])  # Sort based on value
        else:
            # If 5 min values are found, check if current value is smaller than the maximum in min_values
            max_index = max(enumerate(min_values), key=lambda x: x[1][1])[0]  # Index of the maximum value
            if res < min_values[max_index][1]:
                min_values[max_index] = (i, res)  # Replace the maximum value with current value
                min_values.sort(key=lambda x: x[1])  # Resort the list

    return min_values

def sim_search(img, db):
  colors = ("b", "g", "r")
  #db is the descriptor database combining both
  #img is an opened image
  hist_dict = {}
  for i, color in enumerate(colors):
  # Calculate the channel index
    channel_index = colors.index(color)
    # Calculate the histogram for the selected color channel
    hist = cv2.calcHist([img], [channel_index], None, [256], [0, 256])
    hist /= hist.sum()
    hist_dict[color] = hist
  imghist = hist_dict
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  # Calculate the Harris descriptor
  imgharr = cv2.cornerHarris(gray, 2, 3, 0.04)
  harrDs = []
  histDs =[]
  files =[]
  for i in db:
    histD = calculate_hist_d(imghist,db[i]["histogram"])
    harrD = calculate_harris_d(imgharr,db[i]["harris"])
    #dVector = (histD,harrD)
    #distance = calculate_vector_distance_to_origin(dVector)
    files.append(i)
    histDs.append(histD)
    harrDs.append(harrD)
  histV = np.array(histDs)
  harrV = np.array(harrDs)
  histV = normalize_array(histV)
  harrV = normalize_array(harrV)
  return files, find_min_values(files,histV,harrV)



# Load the descriptors dictionary (indexed files)
db = np.load('descriptors.npy', allow_pickle=True).item()
parser = argparse.ArgumentParser(description="Process images and add them to the database")
parser.add_argument("-add", metavar="FILE_PATH", help="Path to the JPG file to add to the database")
parser.add_argument("-rd",action = "store_true", help = "Recalculate descriptors and generate new db")
args = parser.parse_args()

if args.add:
    if args.add.lower().endswith('.jpg') and os.path.exists(args.add):
        addToDb(args.add,db)
        sys.exit()
elif args.rd:
    recalcDescriptors()
    sys.exit()

# Ask the user for the path of the query image
query_image_path = input("Enter the path of the query image: ") #i.e /content/gdrive/MyDrive/italy/Image01.jpg
query_img = cv2.imread(query_image_path)

if query_img is None:
    print("Error: Unable to load the query image.")
else:
    files, res = sim_search(query_img, db)
    top5Files = [files[i] for (i, j) in res]
    similarities = [j for (i, j) in res]
    print("Top 5 similar files:")
    for file, similarity in zip(top5Files, similarities):
        fileN = os.path.join(directory, file)
        if os.name == 'nt':  # Check if the OS is Windows
          fileN = fileN.replace('\\', '/')  # Replace forward slashes with double backslashes
        print("File:", fileN, "Similarity:", similarity)
        similar_img = cv2.imread(fileN)
        plt.imshow(cv2.cvtColor(similar_img, cv2.COLOR_BGR2RGB))
        plt.title(f"Similarity: {similarity}")
        plt.axis('off')
        plt.show()