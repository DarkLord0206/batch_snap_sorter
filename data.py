import cv2
import os
import pandas as pd

dictionary = {}
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
new_path = "./data/cropped"
result = 0
os.makedirs("./data/cropped/train/",exist_ok=True)
os.makedirs("./data/cropped/val/",exist_ok=True)
os.makedirs("./data/cropped/test/",exist_ok=True)
dict = {}
with open("./data/part.txt") as file:
	for line in file:
		(key, value) = line.split()
		dict[key] = value
print(dict)
print(len(dict))
with open("./data/names.txt") as file:
	for line in file:
		(key, value) = line.split()
		curr = "./data/uncropped/" + key
		try:
			img = cv2.imread(curr)
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		except Exception:
			continue
		faces = face_cascade.detectMultiScale(gray, 1.1, 4)  # print(len(faces))
		if len(faces) > 0:
			x, y, w, h = faces[0]
			# cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
			faces = img[y:y + h, x:x + w]
			if dict[key] == "0":
				os.makedirs("./data/cropped/train/" + value, exist_ok=True)
				status = cv2.imwrite(new_path + "/train" + "/" + value + "/" + key, faces)
			if dict[key] == "1":
				os.makedirs("./data/cropped/val/" + value, exist_ok=True)
				status = cv2.imwrite(new_path + "/val" + "/" + value + "/" + key, faces)
			if dict[key] == "2":
				os.makedirs("./data/cropped/test/" + value, exist_ok=True)
				status = cv2.imwrite(new_path + "/test" + "/" + value + "/" + key, faces)
			result = result + status
print(status)
