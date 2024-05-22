from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler
import base64
from PIL import Image
from io import BytesIO


# Convert Image to Base64 
def pil_2_b64(image):
    buff = BytesIO()
    image.save(buff, format="JPEG")
    img_str = base64.b64encode(buff.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{img_str}"

def create_nanollava():   
    chat_handler = Llava15ChatHandler(clip_model_path="nanolava/nanollava-mmproj-f16.gguf")
    llm = Llama(model_path="nanolava/nanollava-text-model-f16.gguf",
                chat_handler=chat_handler,
                n_ctx=2048, # n_ctx should be increased to accomodate the image embedding
                verbose=False
                )
    return llm

vllm = create_nanollava()

uploadedImage = Image.open('image.jpg')
data_uri = pil_2_b64(uploadedImage)
myprompt = 'Describe with details the picture.'
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

completion  =  vllm.create_chat_completion(messages=messages,  
                    stop=["###", "<|endoftext|>"],
                    max_tokens=350,
                    temperature=0.1,
                    repeat_penalty=1.2)
print(completion['choices'][0]['message']['content'])