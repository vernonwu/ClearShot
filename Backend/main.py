from fastapi import FastAPI, File, UploadFile
import cv2
from fastapi.responses import FileResponse
from uvicorn import run
import numpy as np
import time
import subprocess
import os
from putscp import scp_upload_img, scp_download_img,connect_scp
from pydantic import BaseModel
import json
import pickle
from multiprocessing import Queue
import threading

app = FastAPI()


class FeedbackItem(BaseModel):
    imageName: str
    contour: list[list]

def save_pkl(imagename,imagetype,sliderValue,contour):
    with open("feedback/"+imagename+imagetype,'rb') as image_file:
        image_data = image_file.read()
                        
    data = {
        "image":image_data,
        "sliderValue": sliderValue,
        "contour":contour
    }
    with open('feedback/'+imagename+'.pkl','wb') as file:
        pickle.dump(data,file)

def convert_array(arr):
    print(arr)
    a = arr[0][0]
    b = arr[0][1]
    c = arr[1][0]
    d = arr[1][1]
    return np.array([[[a,b],[c,b],[c,d],[a,d]]])

def read_pkl(name):
    with open('feedback/' + name+'.pkl', "rb") as file:
        data = pickle.load(file)

    image_data = data["image"]
    sliderValue = data["sliderValue"]
    contour_data = data["contour"]

    image_array = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    contour_points = np.array(contour_data, np.int32)
    contour_points = convert_array(contour_points)
    cv2.polylines(image, [contour_points], isClosed=True, color=(255, 0, 0), thickness=2)
    print(sliderValue)
    cv2.imshow("Image with Contour", image)
    cv2.waitKey(0)

@app.post("/feedback/")
async def feedback(item:FeedbackItem):
    rname = item.imageName.split('-')[0]
    imagetype = item.imageName.split('-')[-1][3:]
    sliderValue = item.imageName.split('-')[-1][:3]
    save_pkl(rname,imagetype,sliderValue,item.contour)
    print("save {}.pkl".format(rname))
    return {"imageName": item.imageName, "contour": item.contour}

server_on=False
'''
q = Queue()

def multi_scp():
    print("scp",q)
    if server_on:
        while 1:
            if not q.empty():
                scp_upload_img(q.get())

            time.sleep(0.01)
'''
scpcilent = None
if server_on == True:
    scpcilent = connect_scp()
@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    data = await file.read()
    image_np = np.frombuffer(data, np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    # 保存处理后的图像
    rname = file.filename
    name = time.strftime('%Y-%m-%d-%H-%M-%S')+".jpg"
    cv2.imwrite("post_image/"+name, image)
    cv2.imwrite("feedback/"+rname, image)
    if server_on:
        scpcilent.put("C:/Users/Administrator/aisys/fastapi-main/post_image/"+name,"projects/FFTformer/media/pred/input/1/")
        while 1:
            try:
                scpcilent.get("projects/FFTformer/results/Adaptive_fftformer/output/1/"+name,"C:/Users/Administrator/aisys/fastapi-main/get_image/")
            except Exception:
                time.sleep(0.01)
                continue
            else:
                break
        os.remove("post_image/"+name)
        return FileResponse("get_image/"+name)
    else:
        return FileResponse("post_image/image_post2.jpg")

@app.post("/temp/")
async def quick_return(file: UploadFile = File(...)):
    name = file.filename
    rname = name.split('-')[0]
    typename = name.split('.')[-1]
    time.sleep(1)
    return FileResponse("temp/"+rname+'2.'+typename)

@app.get("/get_image/")
async def get_image(image_name: str):
    image_path = f"get_image/{image_name}"  # 图片文件的路径
    return FileResponse(image_path)

@app.get("/favicon.ico")
async def get_favicon():
    return {"file": "favicon.ico"}



if __name__=="__main__":
    run("main:app",host="0.0.0.0",port=8000)
