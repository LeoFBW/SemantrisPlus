from openai import OpenAI  
from dotenv import load_dotenv
import os
load_dotenv()


client = OpenAI(api_key="sk-jttbbxsdepfuvkuaezofijfjispbwcyvnfuereikqkssavpv", base_url="https://api.siliconflow.cn/v1")  
response = client.chat.completions.create(  
    model=os.environ.get('TEST_CUSTOM_MODEL_NAME', 'tencent/Hunyuan-MT-7B'),  
    messages=[  
        {'role': 'user',  
        'content': "Tell me about London"}  
    ],  
    stream=True  
)  

for chunk in response:
    if not chunk.choices:
        continue
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
    if chunk.choices[0].delta.reasoning_content:
        print(chunk.choices[0].delta.reasoning_content, end="", flush=True)