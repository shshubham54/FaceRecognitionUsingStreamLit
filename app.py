#app.py
import streamlit as st
import cv2
from PIL import Image
import numpy as np
import pickle
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import pandas as pd
from datetime import date


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
model = pickle.load(open("data/face_recognition_model.pkl", 'rb'))
out_encoder=pickle.load(open("data/out_encoder.pkl", 'rb'))
facenet_model = load_model("data/facenet_keras.h5")

default_image_size = tuple((160,160))
image_size = 0
#to extract face from a picture
def extract_face(img):
  # Reading the given Image and converting it to Grayscale
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  # Detecting Faces from image using Face_Samples
  faces = face_cascade.detectMultiScale(gray, 1.3, 5)
  global crop_img
  # Iterating for the dimentions of the Detected faces to draw rectangle
  for (x,y,w,h) in faces:
      crop_img = img[y:y+w, x:x+w]
  return crop_img

def convert_image_to_array(image_dir):
    try:
        image = cv2.imread(image_dir)
        if image is not None :
            image = extract_face(image)
            image = cv2.resize(image, default_image_size)   
            return img_to_array(image)
        else :
            return np.array([])
    except Exception as e:
        print(f"Error : {e}")
        return None

def get_embedding(model, face_pixels):
	# scale pixel values
	face_pixels = face_pixels.astype('float32')
	# standardize pixel values across channels (global)
	mean, std = face_pixels.mean(), face_pixels.std()
	face_pixels = (face_pixels - mean) / std
	# transform face into one sample
	samples = np.expand_dims(face_pixels, axis=0)
	# make prediction to get embedding
	yhat = model.predict(samples)
	return yhat[0]

def mark_attendance(name):
  today = date.today()
  d1 = today.strftime("%d%m%Y")
  sheet = pd.read_csv("data/attendance.csv")
  if(d1 not in sheet):
    sheet[d1]=0
  child = 'dev'
  row_index = np.where(sheet['Name']==child)[0][0]
  sheet.at[row_index,d1] = 1
  sheet.to_csv("data/attendance.csv", index=False)
  total_classes = sheet.shape[1]-1
  child_row = sheet.loc[sheet['Name']==child].values[0]
  child_total_attendance= sum(child_row[1:])
  attendance_percentage = (child_total_attendance*100)/total_classes
  return attendance_percentage

def detect_faces(our_image):
  img=convert_image_to_array(our_image)
  #normalization of images
  np_img = np.array(img, dtype=np.float16) / 225.0
  #get embedded image
  embedding = get_embedding(facenet_model, np_img)
  # prediction for the face
  samples = np.expand_dims(embedding, axis=0)
  yhat_class = model.predict(samples)
  yhat_prob = model.predict_proba(samples)
  # get name
  class_index = yhat_class[0]
  class_probability = yhat_prob[0,class_index] * 100
  predict_names = out_encoder.inverse_transform(yhat_class)
  attendance_percentage = mark_attendance(predict_names[0])
  s=(f'You are recognized as: {predict_names[0]} with {round(class_probability,2)}% accuracy and your attendance is updated to: {attendance_percentage}%')
  return s

 #Main function =================================================================================== 
def main():
    """Face Recognition App"""

    st.title("Smart Attendance System")

    html_temp = """
    <body style="background-color:red;">
    <div style="background-color:teal ;padding:10px">
    <h2 style="color:white;text-align:center;">Face Recognition WebApp</h2>
    </div>
    </body>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    image_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])

    if image_file is not None:
        our_image = Image.open(image_file)
        st.text("Original Image")
        st.image(our_image)
        img=np.array(our_image)
        img=Image.fromarray(img)
        #data.save("temp.jpg")
        #img = "/content/temp.jpg"


    if st.button("Mark Attendance"):
        result_img= detect_faces(img)
        st.success(result_img)


if __name__ == '__main__':
    main()
