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
    
# FUNCTION TO LOG ALL CHAT MESSAGES INTO chathistory.txt
def writehistory(text):
    with open('chathistoryPhi3mini.txt', 'a', encoding='utf-8') as f:
        f.write(text)
        f.write('\n')
    f.close()

#AVATARS
av_us = '🧑‍💻'  # './man.png'  #"🦖"  #A single emoji, e.g. "🧑‍💻", "🤖", "🦖". Shortcodes are not supported.
av_ass = "🤖"   #'./robot.png'

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
    st.session_state.chatUImessages = []   

def main():
    st.set_page_config(layout="wide", page_title="AI Whisper Transcriber")
    vllm = create_nanollava()
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
        data_uri = image_to_base64_data_uri(st.session_state.imagefile)
        st.toast('Ready to **CHAT**', icon='📃')        
    if reset_btn:
        resetall()

    while st.session_state.chatimage:
        # Display chat messages from history on app rerun
        for message in st.session_state.chatUImessages:
            if message["role"] == "user":
                with st.chat_message(message["role"],avatar=av_us):
                    st.image(file1.name, width=350)
                    st.markdown(message["content"])
            else:
                with st.chat_message(message["role"],avatar=av_ass):
                    st.markdown(message["content"])
        # Accept user input
        if myprompt := st.chat_input("What is this?"):
            # Add user message to chat history
            messages = [
                        {"role": "system", "content": "You are an assistant who perfectly describes images."},
                        {
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": data_uri }},
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
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                completion  =  vllm.create_chat_completion(messages=messages,  
                                    stop=["###", "<|endoftext|>"],
                                    max_tokens=350,
                                    temperature=0.1,
                                    repeat_penalty=1.2,
                                    stream=True)
                for chunk in completion:
                    if chunk.choices[0].delta.content:
                        full_response += chunk.choices[0].delta.content
                        message_placeholder.markdown(full_response + "🟠")
                message_placeholder.markdown(full_response)
                asstext = f"assistant: {full_response}"
                writehistory(asstext)       
                st.session_state.chatUImessages.append({"role": "assistant", "content": full_response})

    if  not file1:
        message3.warning("  Upload an image", icon='⚠️')



if __name__ == "__main__":
    main()