import requests

# 上传的图片文件路径
image_file = "post_image/image_post.jpg"

# FastAPI 服务器的地址和端口
server_url = "http://127.0.0.1:8000/upload/"

# 使用requests库向服务器发送POST请求
files = {'file': open(image_file, 'rb')}
response = requests.post(server_url, files=files)
with open('saved_image.jpg', 'wb') as f:
        f.write(response.content)
