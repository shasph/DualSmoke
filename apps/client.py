import requests
import os
import zipfile

payload = {'sketch.png':open('tmp/sketch.png', 'rb')}

response = requests.post('http://localhost:8080/upload', files=payload)
print(response.text)
if response.status_code != requests.codes.ok:
    exit(-1)    

response = requests.post('http://localhost:8080/generate')
if response.status_code != requests.codes.ok:
    print(response.text)
    exit(-1)

with open('./tmp/result.zip', 'wb') as fw:
    fw.write(response.content)

with zipfile.ZipFile('./tmp/result.zip') as pass_zip:
    pass_zip.extractall('./tmp')
