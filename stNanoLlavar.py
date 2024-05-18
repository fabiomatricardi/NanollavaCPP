import streamlit as st
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler
import datetime
import os
import time
import base64

@st.cache_resource 
def create_nanollava():   
    chat_handler = Llava15ChatHandler(clip_model_path="nanolava/nanollava-mmproj-f16.gguf")
    llm = Llama(model_path="nanolava/nanollava-text-model-f16.gguf",
                chat_handler=chat_handler,
                n_ctx=2048, # n_ctx should be increased to accomodate the image embedding
                verbose=False
                )

def image_to_base64_data_uri(file_path):
    with open(file_path, "rb") as img_file:
        base64_data = base64.b64encode(img_file.read()).decode('utf-8')
        return f"data:image/png;base64,{base64_data}"
    

if "gentime" not in st.session_state:
    st.session_state.gentime = "**:green[none yet]**"
if "imagefile" not in st.session_state:
    st.session_state.imagefile = ''   
if "keyimagefile" not in st.session_state:
    st.session_state.keyimagefile = 0     
if "chatimage" not in st.session_state:
    st.session_state.chatimage = 0      


def main():
    st.set_page_config(layout="wide", page_title="AI Whisper Transcriber")
    whisper = create_nanollava()
    st.write("# 🎙️✍️ Talk to your Images with nanollava\n\n\n")
    st.markdown('\n---\n', unsafe_allow_html=True)
    st.sidebar.write("## Upload an image :gear:")
    file1=None
    image_btn = st.button('✨ **Start AI Magic**', type='primary')
    def resetall():
        # tutorial to reset the to 0 the file_uploader from 
        # https://discuss.streamlit.io/t/clear-the-file-uploader-after-using-the-file-data/66178/4
        st.session_state.keyimagefile += 1
        st.session_state.chatimage = 0
        st.rerun()
        
    reset_btn = st.button('🧻 **Reset Image**', type='secondary')
    st.markdown('\n\n')
    message1 = st.empty()
    message11 = st.empty()
    message2 = st.empty()
    message3 = st.empty()
    audioplayer = st.empty()
    transcribed = st.empty()

    # Upload the audio file
    file1 = st.sidebar.file_uploader("Upload an image", 
                                     type=["jpg", "png"],accept_multiple_files=False, 
                                     key=st.session_state.keyimagefile)
    gentimetext = st.sidebar.empty()

    if file1:
        st.session_state.chatimage = 1
        st.session_state.imagefile = file1.name
        st.toast('image file selected!', icon='🎉')
        time.sleep(1.2)
        st.toast('Ready to **CHAT**', icon='📃')        
    if reset_btn:
        resetall()

    while st.session_state.chatimage:
        pass

    if  not file1:
        message3.warning("  Upload an image", icon='⚠️')



if __name__ == "__main__":
    main()