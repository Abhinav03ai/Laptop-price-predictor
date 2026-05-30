import streamlit as st
import pandas as pd
import numpy as np
import pickle 

#TITLE
st.title("Predict the price of your Laptop")

lap=pd.read_csv("/Users/abhinavbadwekar/Documents/Documents - Abhinav’s MacBook Air/project vscode 3/cleaned laptop dataset.csv")
pipe=pickle.load(open('RandomForestRegressor.pkl','rb'))


def sol_predict(brand,Model,processor_brand,processor_tier,ram_memory,primary_storage_type,primary_storage_capacity,gpu_brand,gpu_type,display_size,OS,year_of_warranty):
   prediction=pipe.predict(pd.DataFrame(columns=[ 'brand','Model','processor_brand','processor_tier','ram_memory','primary_storage_type','primary_storage_capacity','gpu_brand','gpu_type','display_size','OS','year_of_warranty'],
                                        data=np.array([brand,Model,processor_brand,processor_tier,ram_memory,primary_storage_type,primary_storage_capacity,gpu_brand,gpu_type,display_size,OS,year_of_warranty]).reshape(1, 12)))
   return str(np.round(prediction[0],2))

def main():
   st.write("Fill the details")
   brand = st.selectbox('Brand', lap['brand'].unique())
   Model = st.selectbox('Model',lap['Model'].unique())
   processor_brand = st.selectbox('Processor_brand', lap['processor_brand'].unique())
   processor_tier = st.selectbox('processor_tier', lap['processor_tier'].unique())
   ram_memory = st.selectbox('ram_memory', lap['ram_memory'].unique())
   primary_storage_type = st.selectbox('primary_storage_type', lap['primary_storage_type'].unique())
   primary_storage_capacity = st.selectbox('primary_storage_capacity', lap['primary_storage_capacity'].unique())
   gpu_brand = st.selectbox('gpu_brand', lap['gpu_brand'].unique())
   gpu_type = st.selectbox('gpu_type', lap['gpu_type'].unique())
   display_size = st.selectbox('display_size', lap['display_size'].unique())
   OS = st.selectbox('OS', lap['OS'].unique())
   year_of_warranty = st.selectbox('year_of_warranty', lap['year_of_warranty'].unique())
   prediction =""
   if st.button('Predict'):
      prediction=np.array(sol_predict(brand,Model,processor_brand,processor_tier,ram_memory,primary_storage_type,primary_storage_capacity,gpu_brand,gpu_type,display_size,OS,year_of_warranty))
      st.success(prediction)   


if __name__ =='__main__':
   main()
