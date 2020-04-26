import cv2
import numpy as np

#Init camera
cap=cv2.VideoCapture(0)
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
skip=0
face_data=[]
dataset_path='C:/Users/tanay/Desktop/Face/dataset/'

file_name=input("enter name of person is scanning")

while True:
	ret,frame=cap.read()
	if ret==False:
		continue
	gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	
	faces= face_cascade.detectMultiScale(frame,1.3,5)
	faces=sorted(faces,key=lambda f:f[2]*f[3])
#pick last bcs its largest in area
	for face in faces[-1:]:
		x,y,w,h=face
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)

		#extract region of interest
		offset=10
		face_section=frame[y-offset:y+offset+h,x-offset:x+w+offset]
		face_section=cv2.resize(face_section,(100,100))
		skip+=1
		if(skip%10==0):
			face_data.append(face_section)
			print(len(face_data))

	cv2.imshow("frame",frame)
	cv2.imshow("face_section",face_section)

	key_pressed=cv2.waitKey(1) & 0xFF
	if key_pressed==ord('q'):
		break
#convert our face list array into numpy array
face_data=np.asarray(face_data)
face_data=face_data.reshape((face_data.shape[0],-1))
print(face_data.shape)

#save it inot file system

np.save(dataset_path+file_name+'.npy',face_data)
print("data succesfully saved")

cap.release()
cv2.destroyAllWindows()