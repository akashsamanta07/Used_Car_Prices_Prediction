import pandas as pd
import numpy as np
import joblib
import streamlit as st
st.header("Used Car Prediction ML Model")
cars_data=pd.read_csv("dataset.csv")

def getnum(list):
    return list.split(" ")[0]
cars_data["name"]=cars_data["name"].apply(getnum)
name = st.selectbox('Select Car Brand', cars_data['name'].unique())
year = st.slider('Car Manufactured Year', 2024,1994)
km_driven = st.slider('No of kms Driven', 10,200000)
fuel = st.selectbox('Fuel type', cars_data['fuel'].unique())
seller_type = st.selectbox('Seller  type', cars_data['seller_type'].unique())
transmission = st.selectbox('Transmission type', cars_data['transmission'].unique())
owner = st.selectbox('Seller  type', cars_data['owner'].unique())
mileage = st.slider('Car Mileage(kmpl)', 10,40)
engine = st.slider('Engine CC', 700,5000)
max_power = st.slider('Maximum speed', 50,400)
seats = st.slider('No of Seats', 5,10)
model=joblib.load("modelmain.pkl")
if st.button("Predict"):
    input_data_model = pd.DataFrame(
    [[name,year,km_driven,fuel,seller_type,transmission,owner,mileage,engine,max_power,seats]],
    columns=['name','year','km_driven','fuel','seller_type','transmission','owner','mileage','engine','max_power','seats'])
    input_data_model['owner'].replace(['First Owner', 'Second Owner', 'Third Owner',
       'Fourth & Above Owner', 'Test Drive Car'],
                           [0,1,2,3,4], inplace=True)
    input_data_model['fuel'].replace(['Diesel', 'Petrol', 'LPG', 'CNG'],[0,1,2,3], inplace=True)
    input_data_model['seller_type'].replace(['Individual', 'Dealer', 'Trustmark Dealer'],[0,1,2], inplace=True)
    input_data_model['transmission'].replace(['Manual', 'Automatic'],[0,1], inplace=True)
    input_data_model['name'].replace(['Maruti', 'Skoda', 'Honda', 'Hyundai', 'Toyota', 'Ford', 'Renault',
       'Mahindra', 'Tata', 'Chevrolet', 'Datsun', 'Jeep', 'Mercedes-Benz',
       'Mitsubishi', 'Audi', 'Volkswagen', 'BMW', 'Nissan', 'Lexus',
       'Jaguar', 'Land', 'MG', 'Volvo', 'Daewoo', 'Kia', 'Fiat', 'Force',
       'Ambassador', 'Ashok', 'Isuzu', 'Opel'],
                          [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
                          ,inplace=True)

    car_price = model.predict(input_data_model)
    st.markdown("PREDICT PPRICES:"+ str("%.2f"%car_price[0]))