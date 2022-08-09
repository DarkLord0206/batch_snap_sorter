import cv2
import os
import pandas as pd

dictionary = {}
with open("./data/uncropped/celebs.txt") as file:
	for line in file:
		(key, value) = line.split()
		os.makedirs("./data/cropped/"+value,exist_ok=True)
		dictionary[key] = value
print('\ntext file to dictionary=\n', dictionary)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# Read the input image
new_path = "./data/cropped"
result=0
images = os.listdir("./data/uncropped/images")
for image in images:
	curr = "./data/uncropped/images/" + image
	img = cv2.imread(curr)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.1, 4)  # print(len(faces))
	if len(faces) > 0:
		x, y, w, h = faces[0]
		# cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
		faces = img[y:y + h, x:x + w]
		status = cv2.imwrite(new_path + "/" + dictionary[image] + "/" + image, faces)
		result=result+status
