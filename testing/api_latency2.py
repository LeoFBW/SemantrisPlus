from openai import OpenAI  

client = OpenAI(api_key="sk-jttbbxsdepfuvkuaezofijfjispbwcyvnfuereikqkssavpv", base_url="https://api.siliconflow.cn/v1")  
response = client.chat.completions.create(  
    model='deepseek-ai/DeepSeek-V3.2-Exp',  
    messages=[  
        {'role': 'user',  
        'content': "tell me a story"}  
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