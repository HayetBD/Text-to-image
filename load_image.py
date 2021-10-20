import pandas as pd
import requests
import shutil


df = pd.read_csv('Data/landscapeSet_v4.csv')


#get image from url 
def load_image_set(dataframe):
    for i in range(0,len(dataframe)) :
        image_url =dataframe.iloc[i,2]
        # Open the url image, set stream to True, this will return the stream content.
        resp = requests.get(image_url, stream=True)
        # Open a local file with wb ( write binary ) permission.
        local_file = open('images_sample/_img_'+str(i)+'.jpg', 'wb')
        
        # Set decode_content value to True, otherwise the downloaded image file's size will be zero.
        resp.raw.decode_content = True
        # Copy the response stream raw data to local image file.
        shutil.copyfileobj(resp.raw, local_file)
        # Remove the image url response object.
        del resp



load_image_set(df)

