import main as main
if __name__=="__main__":
    from PIL import Image
    import numpy as np
    test_img="../Project/minion.png"
    test_img=Image.open(test_img)
    k=3
    # string="arinda"
    string="Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum."
    new_img,location=main.encode_enc_lsb_rand(test_img,string[:200]+"$dip$",k)