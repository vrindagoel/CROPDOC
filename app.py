import streamlit as st
import tensorflow as tf
st.set_option('depreciation.showfileUploaderEncoding',false)
@st.cache(allow_output_mutation=True)
def load_model():
    model.tf.keras.models.load_model('/ontent/plant_disease.hdf5')
    return model
lodel=load_model()
st.write("""#plant diseases""")
file=st.file_uploader("Please upload a crop image",type["jpg","png"])
import cv2
from PIL import Image, ImageOps
import numpy as np
def import_and_predict(image_data,model):
    image=ImageOps.fit(image_data,target_size,Image.ANTIALIAS)
    img=np.asarray(image)
    img_reshape=img[np.newaxis,...]
    prediction=model.predict(img_reshape)
    
    return prediction
if file is None:
    st.text("please upload an image file")
else:
    image=Image.open(file)
    st.image(image,use_column_width=True)
    predictions=import_and_predict(image,model)
    with open("class_indices.json") as file:
    class_names = json.load(file)
    string="The crop is: "+class_names[np.argmax(predictions)]
    st.success(string)