import google.generativeai as genai
genai.configure(api_key="AIzaSyDN36bbdZJgUyUsRQJeDRnNHoQpUsuXe60")

for m in genai.list_models():
    print(m.name)