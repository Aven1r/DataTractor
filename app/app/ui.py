import requests
from requests_toolbelt.multipart.encoder import MultipartEncoder
import streamlit as st
import pandas as pd
from io import BytesIO
from datetime import datetime

backend = "http://ml:8001/api/v1/ml/predict"

def process(image_bytes, server_url: str):
    df = pd.DataFrame()
    try:
        m = MultipartEncoder(fields={"file": ("filename", image_bytes, "image/jpeg")})
        r = requests.post(
            server_url, data=m, headers={"Content-Type": m.content_type},
        )
        if r.status_code == 200:
            st.write("Request successful!")
              # or any other desired output
            df = pd.read_csv(BytesIO(r.content))
        else:
            st.write(f"Request failed with status code: {r.status_code}")
    except Exception as e:
        st.write(f"An error occurred: {e}")
        # Print or log the error for debugging
        print(f"Error occurred: {e}")
    return r.content

# Construct UI layout
st.title("DataTrack")


data_file = st.file_uploader("Загрузите данные трактора  CSV файлом")  # Image upload widget
if st.button('Получить предсказания'):
    if data_file is not None:
        df = data_file.read()
        result = process(data_file, backend)
        print('---')
        print(type(result))
        df = pd.read_csv(BytesIO(result), index_col=0)
        st.write(df)
    
        def func(df):
            d = {}
            # Преобразование строки в дату для первого элемента в 'ds'
            start_date = pd.to_datetime(df['ds'].iloc[0])
            df = df.sort_values(by='ds')
            
            for col in ['transmission', 'hydraulics', 'electric', 'engine', 'brake']:
               
                if col == 'engine':
                    condition = (df[col] < 0) | (df[col] > 103)
                elif col == 'electric':
                    condition = (df[col] < 22) | (df[col] > 30)
                elif col == 'hydraulics':
                    condition = (df[col] < 0) | (df[col] > 1)
                elif col == 'transmission':
                    condition = (df[col] < 0) | (df[col] > 1500)
                elif col == 'brake':
                    condition = (df[col] < 0) | (df[col] > 840)
                
               
                first_idx = condition[condition].first_valid_index()
                
                if first_idx is not None:
                    date_of_event = pd.to_datetime(df.loc[first_idx, 'ds'])
                    days_difference = (date_of_event - start_date)
                    d[col] = days_difference
                else:
                    d[col] = 'Аномалий нет'
            
            return d
        dictt = func(df)
        dd = pd.DataFrame([dictt])
        st.write(dd)



        
    else:
        st.write('Загрузите csv файл!')




# if st.button("Get segmentation map"):
    # bytes_df = 

    # col1, col2 = st.columns(2)

    # if input_image:
    #     segments = process(input_image, backend)
    #     original_image = Image.open(input_image).convert("RGB")
    #     segmented_image = Image.open(io.BytesIO(segments.content)).convert("RGB")
    #     col1.header("Original")
    #     col1.image(original_image, use_column_width=True)
    #     col2.header("Segmented")
    #     col2.image(segmented_image, use_column_width=True)

    # else:
    #     # handle case with no image
    #     st.write("Insert an image!")
