import streamlit as st
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler
import datetime
import os
import time
import base64
from PIL import Image
from io import BytesIO

st.set_page_config(layout="wide", page_title="AI Whisper Transcriber")
# Convert Image to Base64 
def pil_2_b64(image):
    buff = BytesIO()
    image.save(buff, format="JPEG")
    img_str = base64.b64encode(buff.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{img_str}"

@st.cache_resource 
def create_nanollava():   
    chat_handler = Llava15ChatHandler(clip_model_path="nanolava/nanollava-mmproj-f16.gguf")
    llm = Llama(model_path="nanolava/nanollava-text-model-f16.gguf",
                chat_handler=chat_handler,
                n_ctx=2048, # n_ctx should be increased to accomodate the image embedding
                verbose=False
                )
    return llm
    
# FUNCTION TO LOG ALL CHAT MESSAGES INTO chathistory.txt
def writehistory(text):
    with open('chathistoryPhi3mini.txt', 'a', encoding='utf-8') as f:
        f.write(text)
        f.write('\n')
    f.close()

#AVATARS
av_us = '🧑‍💻'  # './man.png'  #"🦖"  #A single emoji, e.g. "🧑‍💻", "🤖", "🦖". Shortcodes are not supported.
av_ass = "✨"   #'./robot.png'

if "gentime" not in st.session_state:
    st.session_state.gentime = "**:green[none yet]**"
if "imagefile" not in st.session_state:
    st.session_state.imagefile = ''   
if "keyimagefile" not in st.session_state:
    st.session_state.keyimagefile = 0     
if "chatimage" not in st.session_state:
    st.session_state.chatimage = 0   
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []   
if "chatUImessages" not in st.session_state:
    st.session_state.chatUImessages = [{"role": "assistant", "content": "Hi there! I am here to assist you with this Image. What do you want to know?"}]   
if "uploadedImage" not in st.session_state:
    st.session_state.uploadedImage = '' 
if "data_uri" not in st.session_state:
    st.session_state.data_uri = '' 



vllm = create_nanollava()
st.write("# 🖼️💬 Talk to your Images with nanollava\n\n\n")
st.markdown('\n---\n', unsafe_allow_html=True)
st.sidebar.image('logonanollava.jpg')

file1=None
#image_btn = st.button('✨ **Start AI Magic**', type='primary')
def resetall():
    # tutorial to reset the to 0 the file_uploader from 
    # https://discuss.streamlit.io/t/clear-the-file-uploader-after-using-the-file-data/66178/4
    st.session_state.keyimagefile += 1
    st.session_state.chatimage = 0
    st.session_state.chatUImessages = [{"role": "assistant", "content": "Hi there! I am here to assist you with this Image. What do you want to know?"}]
    st.rerun()
    
reset_btn = st.sidebar.button('🧻✨ **Reset Image** ', type='primary')
st.sidebar.write("## Upload an image :gear:")
st.markdown('\n\n')
message1 = st.sidebar.empty()
message11 = st.sidebar.empty()
message2 = st.empty()
message3 = st.empty()

# Upload the audio file
file1 = st.sidebar.file_uploader("Upload an image", 
                                    type=["jpg", "png"],accept_multiple_files=False, 
                                    key=st.session_state.keyimagefile)
gentimetext = st.sidebar.empty()

if file1:
    st.session_state.chatimage = 1
    st.session_state.imagefile = file1
    st.session_state.uploadedImage = Image.open(st.session_state.imagefile)
    message1.write('image file selected!')
    # https://stackoverflow.com/questions/52411503/convert-image-to-base64-using-python-pil
    # https://huggingface.co/docs/api-inference/detailed_parameters
    st.session_state.data_uri = pil_2_b64(st.session_state.uploadedImage)
    message11.write('Ready to **CHAT**')        
    if reset_btn:
        resetall()

    with st.chat_message("user",avatar=av_us):
        st.image(st.session_state.uploadedImage, width=350)
    # Display chat messages from history on app rerun
    for message in st.session_state.chatUImessages:
        if message["role"] == "user":
            with st.chat_message(message["role"],avatar=av_us):
                st.markdown(message["content"])
        else:
            with st.chat_message(message["role"],avatar=av_ass):
                st.markdown(message["content"])
    # Accept user input
    if myprompt := st.chat_input("What is this?"): #,key=str(datetime.datetime.now())
        # Add user message to chat history
        st.session_state.messages = [
                    {"role": "system", "content": "You are an assistant who perfectly describes images."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": st.session_state.data_uri }},
                            {"type" : "text", "text": myprompt}
                        ]
                    }
                ]
        st.session_state.chatUImessages.append({"role": "user", "content": myprompt})
        # Display user message in chat message container
        with st.chat_message("user", avatar=av_us):
            st.markdown(myprompt)
            usertext = f"user: {myprompt}"
            writehistory(usertext)
            # Display assistant response in chat message container
        with st.chat_message("assistant",avatar=av_ass):
            message_placeholder = st.empty()
            with st.spinner("Thinking..."):
                full_response = ""
                completion  =  vllm.create_chat_completion(messages=st.session_state.messages,  
                                    stop=["###", "<|endoftext|>"],
                                    max_tokens=350,
                                    temperature=0.1,
                                    repeat_penalty=1.2,
                                    #stream=True
                                    )
    #            for chunk in completion:
    #               print(chunk)
    #               if chunk.choices[0].delta.content:
    #                   full_response += chunk.choices[0].delta.content
    #                    message_placeholder.markdown(full_response + "🟠")
            message_placeholder.markdown(completion['choices'][0]['message']['content'])
            print(completion['choices'][0]['message']['content'])
            asstext = f"assistant: {completion['choices'][0]['message']['content']}"
            writehistory(asstext)       
            st.session_state.chatUImessages.append({"role": "assistant", "content": completion['choices'][0]['message']['content']})

if  not file1:
    message3.warning("  Upload an image", icon='⚠️')

