import cv2
import os

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# Read the input image
new_path = "./data/cropped"
classes = os.listdir("./data/uncropped")
for actor in classes:
	os.mkdir(new_path+"/"+actor)
	curr = "./data/uncropped/" + actor
	files = os.listdir(curr)
	for name in files:
		# print(name)
		curr_2 = curr + "/" + name
		print(curr_2)
		img = cv2.imread(curr_2)
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		faces = face_cascade.detectMultiScale(gray, 1.1, 4)
		# print(len(faces))
		if len(faces)>0:
			x, y, w, h = faces[0]
			# cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
			faces = img[y:y + h, x:x + w]
			status=cv2.imwrite(new_path + "/" + actor + "/" + name, faces)
			print(status)
