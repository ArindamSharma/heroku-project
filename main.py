try:

    from enum import Enum
    from io import BytesIO, StringIO
    from typing import Union
    from PIL import Image
    import pandas as pd
    import streamlit as st
    import cv2
    import numpy as np

except Exception as e:
    print(e)

STYLE = """
<style>
img {
    max-width: 100%;
}
</style>
"""




# Convert encoding data into 8-bit binary
# form using ASCII value of characters
def genData(data):

        # list of binary codes
        # of given data
        newd = []

        for i in data:
            newd.append(format(ord(i), '08b'))
        return newd

# Pixels are modified according to the
# 8-bit binary data and finally returned
def modPix(pix, data):

    datalist = genData(data)
    lendata = len(datalist)
    imdata = iter(pix)

    for i in range(lendata):

        # Extracting 3 pixels at a time
        pix = [value for value in imdata.__next__()[:3] +
                                imdata.__next__()[:3] +
                                imdata.__next__()[:3]]

        # Pixel value should be made
        # odd for 1 and even for 0
        for j in range(0, 8):
            if (datalist[i][j] == '0' and pix[j]% 2 != 0):
                pix[j] -= 1

            elif (datalist[i][j] == '1' and pix[j] % 2 == 0):
                if(pix[j] != 0):
                    pix[j] -= 1
                else:
                    pix[j] += 1
                # pix[j] -= 1

        # Eighth pixel of every set tells
        # whether to stop ot read further.
        # 0 means keep reading; 1 means thec
        # message is over.
        if (i == lendata - 1):
            if (pix[-1] % 2 == 0):
                if(pix[-1] != 0):
                    pix[-1] -= 1
                else:
                    pix[-1] += 1

        else:
            if (pix[-1] % 2 != 0):
                pix[-1] -= 1

        pix = tuple(pix)
        yield pix[0:3]
        yield pix[3:6]
        yield pix[6:9]

def encode_enc(newimg, data):
    w = newimg.size[0]
    (x, y) = (0, 0)

    for pixel in modPix(newimg.getdata(), data):

        # Putting modified pixels in the new image
        newimg.putpixel((x, y), pixel)
        if (x == w - 1):
            x = 0
            y += 1
        else:
            x += 1

# Encode data into image
def encode(image):
    #img = input("Enter image name(with extension) : ")
    #image = Image.open(img, 'r')

    data=st.text_input("Enter data to be encoded ")

    if(st.button('Submit')):
        if (len(data) == 0):
            #raise ValueError('Data is empty')
            st.error("Data is empty")
        else:
            result = data
            st.success(result)
            newimg = image.copy()
            encode_enc(newimg, data)
            st.image(newimg, channels="BGR")


    #new_img_name = input("Enter the name of new image(with extension) : ")
    #newimg.save(new_img_name, str(new_img_name.split(".")[1].upper()))

# Decode the data in the image
def decode(image):
    #img = input("Enter image name(with extension) : ")
    #image = Image.open(img, 'r')

    data = ''
    imgdata = iter(image.getdata())

    while (True):
        pixels = [value for value in imgdata.__next__()[:3] +
                                imgdata.__next__()[:3] +
                                imgdata.__next__()[:3]]

        # string of binary data
        binstr = ''

        for i in pixels[:8]:
            if (i % 2 == 0):
                binstr += '0'
            else:
                binstr += '1'

        data += chr(int(binstr, 2))

        if (pixels[-1] % 2 != 0):
            return data



def run():

    #st.markdown("<h1 style='text-align: center; color: red;'>Some title</h1>", unsafe_allow_html=True)
    st.set_page_config(layout="wide")

    info ="""
    # Image Steganography
    ## DIP Project
    ### Group No. 41






    * Firoz Mohammad CED17I017
    * Arindam Sharma CED17I022
    * Vaibhav Singhal CED17I040
    """
    st.sidebar.markdown(info, unsafe_allow_html=True)
    #st.sidebar.markdown("Arindam Sharma CED17I022")
    #st.sidebar.markdown("Vaibhav Singhal CED17I040")





    fileTypes = ["png"]
    st.markdown(STYLE, unsafe_allow_html=True)
    choice = st.radio('Choose',["Encode","Decode"])
    if(choice=="Encode"):
        file = st.file_uploader("Upload Cover Image", type=fileTypes)
        show_file = st.empty()
        if not file:
            show_file.info("Please upload a file of type: " + ", ".join(["png"]))
            return

        ####
        #file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        #opencv_image = cv2.imdecode(file_bytes, 1)

        im = Image.open(BytesIO(file.read()))

        encode(im)
        ####

        content = file.getvalue()
        if isinstance(file, BytesIO):
            show_file.image(file)

        file.close()

    elif(choice=="Decode"):
        file = st.file_uploader("Upload Encoded Image", type=fileTypes)
        show_file = st.empty()
        if not file:
            show_file.info("Please upload a file of type: " + ", ".join(["png"]))
            return

        ####
        #file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        #opencv_image = cv2.imdecode(file_bytes, 1)

        im = Image.open(BytesIO(file.read()))

        data=decode(im)
        if(st.button('Decode')):
            st.success(data)
        ####

        content = file.getvalue()
        if isinstance(file, BytesIO):
            show_file.image(file)

        file.close()





run()
