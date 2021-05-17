try:
    from io import BytesIO
    from PIL import Image
    import streamlit as st
    import numpy as np
    import base64

except Exception as e:
    print(e)

STYLE = """
<style>
img {
    max-width: 100%;
}
</style>
"""

astyle = """
display: inline;
width: 200px;
height: 40px;
background: #F63366;
padding: 9px;
margin: 8px;
text-align: center;
vertical-align: center;
border-radius: 5px;
color: white;
line-height: 25px;
text-decoration: none;
"""


# Some Usefull Function ============================

def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])

	# return the MSE, the lower the error, the more "similar" the two images are
	return err

def get_image_download_link(filename,img):
	buffered = BytesIO()
	img.save(buffered, format="PNG")
	img_str = base64.b64encode(buffered.getvalue()).decode()
	href = '<a href="data:file/png;base64,'+img_str+'" download='+filename+' style="'+astyle+'" target="_blank">Download Image</a>'

	return href

def get_key_download_link(filename,key):
    buffered = BytesIO()
    key.dump(buffered)
    key_str = base64.b64encode(buffered.getvalue()).decode()
    href = '<a href="data:file/pkl;base64,'+key_str+'" download='+filename+' style="'+astyle+'" target="_blank">Download Key</a>'
    return href

# Algo 1 =======================================

# Pixels are modified according to the 8-bit binary data and finally returned
def modPix(pix, data):

    datalist =[format(ord(i), '08b') for i in data]
    lendata = len(datalist)
    imdata = iter(pix)

    for i in range(lendata):

        # Extracting 3 pixels at a time
        pix = [value for value in imdata.__next__()[:3] + imdata.__next__()[:3] + imdata.__next__()[:3]]

        # Pixel value should be made odd for 1 and even for 0 ,pix is one channel of a pixel
        for j in range(0, 8):
            if (datalist[i][j] == '0'):
                pix[j] &= ~(1<<0)

            elif (datalist[i][j] == '1'):
                pix[j] |= (1<<0)


        # Eighth pixel of every set tells whether to stop ot read further.# 0 means keep reading; 1 means the message is over.
        if (i == lendata - 1):
            pix[-1] |= (1<<0)

        else:
            pix[-1] &= ~(1<<0)


        pix = tuple(pix)
        yield pix[0:3] #pixel 1
        yield pix[3:6] #pixel 2
        yield pix[6:9] #pixel 3

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
def encode(filename,image,bytes):

    global c1,c2


    data=c1.text_area("Enter data to be encoded ",max_chars=bytes)

    if(c1.button('Encode',key="1")):
        if (len(data) == 0):
            c1.error("Data is empty")
        else:
            c2.markdown('#')
            result = "The given data is encoded in the given cover image"
            c2.success(result)
            c2.markdown('####')
            c2.markdown("#### Encoded Image")
            c2.markdown('######')

            newimg = image.copy()
            encode_enc(newimg, data)
            c2.image(newimg, channels="BGR")

            filename='encoded_'+filename

            image_np = np.array(image)
            newimg_np = np.array(newimg)
            MSE=mse(image_np,newimg_np)
            msg="MSE: "+str(MSE)
            c2.warning(msg)
            c2.markdown("#")
            c2.markdown(get_image_download_link(filename,newimg), unsafe_allow_html=True)

# Decode the data in the image
def decode(image):

    data = ''
    imgdata = iter(image.getdata())

    while (True):
        pixels = [value for value in imgdata.__next__()[:3] + imgdata.__next__()[:3] + imgdata.__next__()[:3]]

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

# Algo 2 =======================================

# Pixels are modified according to the 8-bit binary data and finally returned
def modPix_rand(img,data,location):

    key=iter(location)
    datalist =[format(ord(i), '08b') for i in data]
    lendata = len(datalist)

    for i in range(lendata):
        # Extracting 3 pixels at a time
        x1,y1=next(key)
        x2,y2=next(key)
        x3,y3=next(key)
        x1,y1=int(x1),int(y1)
        x2,y2=int(x2),int(y2)
        x3,y3=int(x3),int(y3)

        pix = [value for value in img.getpixel((x1,y1))[:3] +img.getpixel((x2,y2))[:3] +img.getpixel((x3,y3))[:3]]
        # Pixel value should be made odd for 1 and even for 0
        #pix is one channel of a pixel
        for j in range(0, 8):
            if (datalist[i][j] == '0'):
                pix[j] &= ~(1<<0)

            elif (datalist[i][j] == '1'):
                pix[j] |= (1<<0)

        # Eighth pixel of every set tells whether to stop ot read further.
        # 0 means keep reading; 1 means the message is over.
        if (i == lendata - 1):
            pix[-1] |= (1<<0)

        else:
            pix[-1] &= ~(1<<0)

        pix = tuple(pix)

        img.putpixel((x1,y1),pix[0:3]) #pixel 1
        img.putpixel((x2,y2),pix[3:6]) #pixel 2
        img.putpixel((x3,y3),pix[6:9]) #pixel 3

    return img

def encode_enc_rand(newimg, data):

    w,h = newimg.size

    location = np.array([[(j,i) for i in range(h)] for j in range(w)]).reshape(-1,2)
    np.random.shuffle(location)
    loc1=location.copy()

    return modPix_rand(newimg,data,loc1),location[:len(data)*3]

# Encode data into image
def encode_rand(filename,img,bytes):

    global c1,c2


    data=c1.text_area("Enter data to be encoded ",max_chars=bytes)

    if(c1.button('Encode',key="5")):
        if (len(data) == 0):
            c1.error("Data is empty")
        else:
            c2.markdown('#')
            result = "The given data is encoded in the given cover image"
            c2.success(result)
            c2.markdown('####')
            c2.markdown("#### Encoded Image")
            c2.markdown('######')

            newimg = img.copy()
            newimg,key=encode_enc_rand(newimg, data)
            c2.image(newimg, channels="BGR")
            filename='encoded_'+filename

            image_np = np.array(img)
            newimg_np = np.array(newimg)
            MSE=mse(image_np,newimg_np)
            msg="MSE: "+str(MSE)
            c2.warning(msg)
            c2.markdown("#")
            filename1 = 'key.pkl'
            c2.markdown(get_image_download_link(filename,newimg)+" "+get_key_download_link(filename1,key), unsafe_allow_html=True)

# Decode the data in the image
def decode_rand(img,location):
    key=iter(location)
    data = ''
    ir=0
    while (True):

        # Extracting 3 pixels at a time
        x1,y1=next(key)
        x2,y2=next(key)
        x3,y3=next(key)
        x1,y1=int(x1),int(y1)
        x2,y2=int(x2),int(y2)
        x3,y3=int(x3),int(y3)

        pixels = [value for value in img.getpixel((x1,y1))[:3] +img.getpixel((x2,y2))[:3] +img.getpixel((x3,y3))[:3]]
        # string of binary data
        ir+=1
        binstr = ''

        for i in pixels[:8]:
            if (i % 2 == 0):
                binstr += '0'
            else:
                binstr += '1'

        data += chr(int(binstr, 2))

        if (pixels[-1] % 2 != 0):
            return data

# Algo 3 =======================================

def encode_enc_lsb(img,data,k):

    datalist = "".join([format(ord(i), '08b') for i in data])
    rem=len(datalist)%(3*k)
    if(rem!=0):
        datalist+='0'*(3*k-rem)
    lendata = len(datalist)
    w = img.size[0]
    (x, y) = (0, 0)

    for i in range(0,lendata,k*3):
        kbit_r,kbit_g,kbit_b=datalist[i:i+k],datalist[i+k:i+2*k],datalist[i+2*k:i+3*k] 
        r,g,b=img.getpixel((x,y))[:3]
        # -------------
        # rbg modified = pix
        r=((r>>k)<<k)+int(kbit_r,2)
        g=((g>>k)<<k)+int(kbit_g,2)
        b=((b>>k)<<k)+int(kbit_b,2)
        # -----------
        img.putpixel((x,y),(r,g,b))
        if (x == w - 1):
            x = 0
            y += 1
        else:
            x += 1

    return img

# Encode data into image
def encode_lsb(filename,image,k,bytes):

    data=d1.text_area("Enter data to be encoded ",max_chars=bytes)

    if(d1.button('Encode',key="3")):
        if (len(data) == 0):
            d1.error("Data is empty")
        else:
            d2.markdown('##')
            result = "The given data is encoded in the given cover image"
            d2.success(result)
            d2.markdown('####')
            d2.markdown("#### Encoded Image")
            d2.markdown('######')

            newimg = image.copy()

            encode_enc_lsb(newimg, data+"$dip$",k)
            d2.image(newimg, channels="BGR")

            filename='encoded_'+filename

            image_np = np.array(image)
            newimg_np = np.array(newimg)
            MSE=mse(image_np,newimg_np)
            msg="MSE: "+str(MSE)
            d2.warning(msg)
            d2.markdown("#")
            d2.markdown(get_image_download_link(filename,newimg), unsafe_allow_html=True)

# Decode the data in the image
def decode_lsb(img,k): 
    global d1
    w,h=img.size
    x,y=0,0
    rawdata = ''
    while(x<w and y<h):
        r,g,b=img.getpixel((x,y))[:3]
        rawdata+=format(r, '08b')[-k:]
        rawdata+=format(g, '08b')[-k:]
        rawdata+=format(b, '08b')[-k:]
        if (x == w - 1):
            x = 0
            y += 1
        else:
            x += 1

    # print(rawdata)
    sec_key=""
    for i in "$dip$":
        sec_key+=format(ord(i), '08b')

    index=rawdata.find(sec_key) # $dip$ is eomsg
    if(index==-1):
        # raise Error
        d1.error("Decoding Failed")
        return None

    # print(index)
    rawdata=rawdata[:index]

    data=''
    for i in range(0,len(rawdata),8):
        data+=chr(int(rawdata[i:i+8], 2))

    return data

# Algo 4 =======================================

def encode_enc_lsb_rand(img,data,k):
    w,h = img.size

    location = np.array([[(j,i) for i in range(h)] for j in range(w)]).reshape(-1,2)
    np.random.shuffle(location)
    key=iter(location)
    datalist = "".join([format(ord(i), '08b') for i in data])
    rem=len(datalist)%(3*k)
    if(rem!=0):
        datalist+='0'*(3*k-rem)
    lendata = len(datalist)

    for i in range(0,lendata,k*3):
        x,y=next(key)
        #print(x,y)
        x,y=int(x),int(y)
        kbit_r,kbit_g,kbit_b=datalist[i:i+k],datalist[i+k:i+2*k],datalist[i+2*k:i+3*k] 
        r,g,b=img.getpixel((x,y))[:3]
        # -------------
        # rbg modified = pix
        r=((r>>k)<<k)+int(kbit_r,2)
        g=((g>>k)<<k)+int(kbit_g,2)
        b=((b>>k)<<k)+int(kbit_b,2)

        img.putpixel((x,y),(r,g,b))

    index=np.where(location==next(key))[0][0]
    return img,location[:index]

# Encode data into image
def encode_lsb_rand(filename,image,k,bytes):

    data=d1.text_area("Enter data to be encoded ",max_chars=bytes)

    if(d1.button('Encode',key="7")):
        if (len(data) == 0):
            d1.error("Data is empty")
        else:
            d2.markdown('##')
            result = "The given data is encoded in the given cover image"
            d2.success(result)
            d2.markdown('####')
            d2.markdown("#### Encoded Image")
            d2.markdown('######')

            newimg = image.copy()
            newimg,key=encode_enc_lsb_rand(newimg, data+"$dip$",k)
            d2.image(newimg, channels="BGR")

            filename='encoded_'+filename

            image_np = np.array(image)
            newimg_np = np.array(newimg)
            MSE=mse(image_np,newimg_np)
            msg="MSE: "+str(MSE)
            d2.warning(msg)
            d2.markdown("#")
            filename1 = 'key.pkl'
            d2.markdown(get_image_download_link(filename,newimg)+" "+get_key_download_link(filename1,key), unsafe_allow_html=True)

# Decode the data in the image
def decode_lsb_rand(img,k,location): 
    global d1
    rawdata = ''
    for x,y in location:
        x,y=int(x),int(y)
        r,g,b=img.getpixel((x,y))[:3]
        rawdata+=format(r, '08b')[-k:]
        rawdata+=format(g, '08b')[-k:]
        rawdata+=format(b, '08b')[-k:]

    sec_key=""
    for i in "$dip$":
        sec_key+=format(ord(i), '08b')

    index=rawdata.find(sec_key) # $dip$ is eomsg
    if(index==-1):
        d1.error("Decoding Failed")
        return None

    rawdata=rawdata[:index]
    data=''
    for i in range(0,len(rawdata),8):
        data+=chr(int(rawdata[i:i+8], 2))

    return data


def main():

    global c1,c2,d1,d2

    st.set_page_config(page_title="DIP Project",initial_sidebar_state="expanded",layout="wide")
    hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    st.sidebar.title("Digital Image Processing Project")

    md="![1](https://miro.medium.com/max/2560/1*dQyfOpFWmSxrmdOcQgW6OQ.jpeg)"
    st.sidebar.markdown(md, unsafe_allow_html=True)

    info ="""
    # Image Steganography
    Steganography is the study and practice of concealing information within objects in such a way that it deceives the viewer as if there is no information hidden within the object. It is hiding information in plain sight, such that only the intended recipient would get to see it.
    """
    st.sidebar.markdown(info, unsafe_allow_html=True)
    st.sidebar.subheader("Choose Algorithm: ")
    algo = st.sidebar.radio("",["Algorithm 1","k-LSB Algorithm","Randomized Algorithm 1","Randomized k-LSB Algorithm"])

    info ="""
    ## Group No. 41
    * Firoz Mohammad CED17I017
    * Arindam Sharma CED17I022
    * Vaibhav Singhal CED17I040
    """
    st.sidebar.markdown(info, unsafe_allow_html=True)

    fileTypes = ["png","jpg"]
    fileTypes1 = ["pkl"]
    st.markdown(STYLE, unsafe_allow_html=True)


    if(algo=="Algorithm 1"):
        st.subheader("Algorithm 1")
        st.write("3 pixels (3*3 channels = 9 values) are used to encode one ASCII character. LSBs of first 8 values encode the ASCII in binary format and the LSB of the 9th value is used to represent whether it is end of message.")
        choice = st.radio('Choose',["Encode","Decode"])
        if(choice=="Encode"):
            c1, c2 = st.beta_columns(2)
            file = c1.file_uploader("Upload Cover Image", type=fileTypes, key="fu1")
            show_file = c1.empty()
            if not file:
                show_file.info("Please upload a file of type: " + ", ".join(["png","jpg"]))
                return

            im = Image.open(BytesIO(file.read()))
            filename=file.name
            w,h = im.size
            bytes=(w*h)//3
            c1.info("Max data size: "+str(bytes)+" Bytes")
            encode(filename,im,bytes)

            content = file.getvalue()
            if isinstance(file, BytesIO):
                show_file.image(file)

            file.close()

        elif(choice=="Decode"):
            file = st.file_uploader("Upload Encoded Image", type=fileTypes, key="fu2")
            show_file = st.empty()
            if not file:
                show_file.info("Please upload a file of type: " + ", ".join(["png","jpg"]))
                return

            im = Image.open(BytesIO(file.read()))

            data=decode(im)

            if(st.button('Decode',key="4")):
                st.subheader("Decoded Text")
                st.write(data)

            content = file.getvalue()
            if isinstance(file, BytesIO):
                show_file.image(file)

            file.close()

    elif(algo=="k-LSB Algorithm"):
        st.subheader("k-LSB Algorithm")

        d1, d2 = st.beta_columns(2)
        choice = d1.radio('Choose',["Encode","Decode"])
        k = d2.slider("K bit encoding", min_value=1, max_value=8)

        if(choice=="Encode"):
            file = d1.file_uploader("Upload Cover Image", type=fileTypes, key="fu3")
            show_file = d1.empty()
            if not file:
                show_file.info("Please upload a file of type: " + ", ".join(["png","jpg"]))
                return

            im = Image.open(BytesIO(file.read()))
            filename=file.name
            w,h = im.size
            bits=(w*h*3*k)-40 #sub 40 for $dip$
            bytes=(bits)//8
            d1.info("Max data size: "+str(bytes)+" Bytes")
            encode_lsb(filename,im,k,bytes)

            content = file.getvalue()
            if isinstance(file, BytesIO):
                show_file.image(file)

            file.close()

        elif(choice=="Decode"):
            file = d1.file_uploader("Upload Encoded Image", type=fileTypes, key="fu4")
            show_file = d1.empty()
            if not file:
                show_file.info("Please upload a file of type: " + ", ".join(["png","jpg"]))
                return

            im = Image.open(BytesIO(file.read()))

            if(d1.button('Decode',key="2")):
                data=decode_lsb(im,k)
                if(data!=None):
                    d1.subheader("Decoded Text")
                    d1.write(data)

            content = file.getvalue()
            if isinstance(file, BytesIO):
                show_file.image(file)

            file.close()

    elif(algo=="Randomized Algorithm 1"):
        st.subheader("Randomized Algorithm 1")
        st.write("3 pixels (3*3 channels = 9 values) are used to encode one ASCII character. LSBs of first 8 values encode the ASCII in binary format and the LSB of the 9th value is used to represent whether it is end of message.")
        choice = st.radio('Choose',["Encode","Decode"])
        if(choice=="Encode"):
            c1, c2 = st.beta_columns(2)
            file = c1.file_uploader("Upload Cover Image", type=fileTypes, key="fu5")
            show_file = c1.empty()
            if not file:
                show_file.info("Please upload a file of type: " + ", ".join(["png","jpg"]))
                return

            im = Image.open(BytesIO(file.read()))
            filename=file.name
            w,h = im.size
            bytes=(w*h)//3
            c1.info("Max data size: "+str(bytes)+" Bytes")
            encode_rand(filename,im,bytes)

            content = file.getvalue()
            if isinstance(file, BytesIO):
                show_file.image(file)

            file.close()

        elif(choice=="Decode"):
            d1,d2 = st.beta_columns(2)
            file = d1.file_uploader("Upload Encoded Image", type=fileTypes, key="fu6")
            keyfile = d2.file_uploader("Upload Key", type=fileTypes1, key="fukey")
            show_file = d1.empty()
            if not file:
                show_file.info("Please upload a file of type: " + ", ".join(["png","jpg"]))
                return
            if not keyfile:
                return

            im = Image.open(BytesIO(file.read()))
            location=np.load(BytesIO(keyfile.read()),allow_pickle=True)
            loc2=location.copy()

            data=decode_rand(im,location)

            if(d1.button('Decode',key="6")):
                d1.subheader("Decoded Text")
                d1.write(data)

            content = file.getvalue()
            if isinstance(file, BytesIO):
                show_file.image(file)

            file.close()

    elif(algo=="Randomized k-LSB Algorithm"):
        st.subheader("Randomized k-LSB Algorithm")

        d1, d2 = st.beta_columns(2)
        choice = d1.radio('Choose',["Encode","Decode"])
        k = d2.slider("K bit encoding", min_value=1, max_value=8)

        if(choice=="Encode"):
            file = d1.file_uploader("Upload Cover Image", type=fileTypes, key="fu3")
            show_file = d1.empty()
            if not file:
                show_file.info("Please upload a file of type: " + ", ".join(["png","jpg"]))
                return

            im = Image.open(BytesIO(file.read()))
            filename=file.name
            w,h = im.size
            bits=(w*h*3*k)-40 #sub 40 for $dip$
            bytes=(bits)//8
            d1.info("Max data size: "+str(bytes)+" Bytes")
            encode_lsb_rand(filename,im,k,bytes)

            content = file.getvalue()
            if isinstance(file, BytesIO):
                show_file.image(file)

            file.close()

        elif(choice=="Decode"):
            file = d1.file_uploader("Upload Encoded Image", type=fileTypes, key="fu4")
            keyfile = d2.file_uploader("Upload Key", type=fileTypes1, key="fukey")
            show_file = d1.empty()
            if not file:
                show_file.info("Please upload a file of type: " + ", ".join(["png","jpg"]))
                return
            if not keyfile:
                return

            im = Image.open(BytesIO(file.read()))
            location=np.load(BytesIO(keyfile.read()),allow_pickle=True)
            loc2=location.copy()

            if(d1.button('Decode',key="8")):
                data=decode_lsb_rand(im,k,location)
                if(data!=None):
                    d1.subheader("Decoded Text")
                    d1.write(data)

            content = file.getvalue()
            if isinstance(file, BytesIO):
                show_file.image(file)

            file.close()

if __name__=="__main__":
    main()
