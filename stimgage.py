import streamlit as st
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler
import datetime
import os
import time
import base64
from PIL import Image
from io import BytesIO


# Convert Image to Base64 
def im_2_b64(image):
    buff = BytesIO()
    image.save(buff, format="JPEG")
    img_str = base64.b64encode(buff.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{img_str}"

if "keyimagefile" not in st.session_state:
    st.session_state.keyimagefile = 0   
if "chatimage" not in st.session_state:
    st.session_state.chatimage = 0  
if "uploadedImage" not in st.session_state:
    st.session_state.uploadedImage = ''  

def main():
    st.set_page_config(layout="wide", page_title="AI Whisper Transcriber")
    st.write("# 🎙️✍️ Talk to your Images with nanollava\n\n\n")
    st.markdown('\n---\n', unsafe_allow_html=True)
    st.sidebar.write("## Upload an image :gear:")
    file1=None
    image_btn = st.button('✨ **Start AI Magic**', type='primary')
    reset_btn = st.button('🧻 **Reset Image**', type='secondary')
    message3 = st.empty()
    file1 = st.sidebar.file_uploader("Upload an image", 
                                     type=["jpg", "png"],accept_multiple_files=False, 
                                     key=st.session_state.keyimagefile)
    if file1:    
        st.image(file1)
        st.session_state.uploadedImage = Image.open(file1)
        st.write(im_2_b64(st.session_state.uploadedImage))

    def resetall():
        # tutorial to reset the to 0 the file_uploader from 
        # https://discuss.streamlit.io/t/clear-the-file-uploader-after-using-the-file-data/66178/4
        st.session_state.keyimagefile += 1
        st.session_state.chatimage = 0
        st.rerun()
    if reset_btn:
        resetall()    

    if  not file1:
        message3.warning("  Upload an image", icon='⚠️')



if __name__ == "__main__":
    main()