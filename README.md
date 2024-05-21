# NanoLlava is the tiniest Visual Language Model

Learn how to run it with Llama.cpp

<img src='https://github.com/fabiomatricardi/NanollavaCPP/raw/main/logonanollavaSocial.png' height=200>

---

### Dependencies
- create a virtual environment
- install the following packages:
```
pip install streamlit==1.34.0 llama-cpp-python==0.2.75
```

download in a local directory called `nanolava` the 2 files from:<br>
https://huggingface.co/abetlen/nanollava-gguf
- nanollava-mmproj-f16.gguf
- nanollava-text-model-f16.gguf

### Run the interface

from the terminal with the `venv` activated:
```
streamlit run st-NanoLlava.py
```

<img src='https://github.com/fabiomatricardi/NanollavaCPP/raw/main/nanollavaStreamlit.gif' width=800>
<br><br>










