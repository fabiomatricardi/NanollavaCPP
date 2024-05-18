from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler
from llama_cpp.llama_chat_format import MoondreamChatHandler
import datetime


"""
https://huggingface.co/sroecker/moondream2-GGUF for the GGUF
https://huggingface.co/moondream/moondream2-gguf/tree/main for mmprj


Source/inspiration:
https://www.markhneedham.com/blog/2024/02/04/llava-large-multi-modal-model-v1.5-v1.6/

Extracting code from an image - "Extract the code from this image"
Reading a banner image - "What text is written on this image?"
Captioning an image -   "Create a caption for this image" / 
                        "Create a creative caption for this image"
                        "write a captivating narration to caption this image."
Understanding a diagram - "Can you describe this diagram?"
"""
print('Loading the model...')
chat_handler = MoondreamChatHandler(clip_model_path="moondream2/moondream2-mmproj-f16.gguf")
llm = Llama(
  model_path="moondream2/moondream2-text-model.Q5_K_M.gguf",
  chat_handler=chat_handler,
  n_ctx=2048, # n_ctx should be increased to accomodate the image embedding
  verbose=False
)

print('Select an image to chat with')
import easygui   #https://easygui.readthedocs.io/en/master/api.html
file_path = easygui.fileopenbox(filetypes = ["*.png","*.jpg"])
print(f'Loaded image - {file_path}')

print('converting the image to base64...')
import base64

def image_to_base64_data_uri(file_path):
    with open(file_path, "rb") as img_file:
        base64_data = base64.b64encode(img_file.read()).decode('utf-8')
        return f"data:image/png;base64,{base64_data}"

# Replace 'file_path.png' with the actual path to your PNG file
# file_path = 'image3_2.PNG' - replaced by EasyGUI loading
data_uri = image_to_base64_data_uri(file_path)

print('Creating messages for chat completion')
messages = [
        {
            "role": "user",
            "content": [
                {"type" : "text", "text": "What's in this image?"},
                {"type": "image_url", "image_url": {"url": data_uri } }

            ]
        }
    ]


start = datetime.datetime.now()
print('Running inference...')
response = llm.create_chat_completion(messages=messages,  
                                    stop=["###", "<|endoftext|>"],
                                    max_tokens=350,
                                    temperature=0.1,
                                    repeat_penalty=1.2)
delta = datetime.datetime.now() - start

print(response["choices"][0]["message"]["content"]) #["choices"][0]["message"]["content"]
print('---')
print(f'Generated in {delta}')
messages.append([{'role': 'assistant', 
                  'content': response["choices"][0]["message"]["content"]}])
while True:
    print('Type another question, or  exit!  to quit')
    userq = input('User: ')
    if userq.lower() == 'exit!':
        print('bye bye')
        break
    else:
        messages = [
                {
                    "role": "user",
                    "content": [
                        {"type" : "text", "text": userq},
                        {"type": "image_url", "image_url": {"url": data_uri } }

                    ]
                }
            ]        
        start = datetime.datetime.now()
        print('Running inference...')
        response = llm.create_chat_completion(messages=messages,  
                                            stop=["###", "<|endoftext|>"],
                                            max_tokens=350,
                                            temperature=0.1,
                                            repeat_penalty=1.2)
        delta = datetime.datetime.now() - start        
        print(response["choices"][0]["message"]["content"]) #["choices"][0]["message"]["content"]
        print('---')
        print(f'Generated in {delta}')
        messages.append([{'role': 'assistant', 
                        'content': response["choices"][0]["message"]["content"]}])