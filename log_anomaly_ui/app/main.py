# from typing import Union

# from fastapi import FastAPI

# app = FastAPI()

# @app.get("/")
# def read_root():
#     print("Hello World")
#     return {"Hello": "World"}


# @app.get("/items/{item_id}")
# def read_item(item_id: int, q: Union[str, None] = None):
#     return {"item_id": item_id, "q": q}

from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import shutil, os
from .config import UPLOAD_DIR
# from deeplog import train
# from data_process_pred import 
# from models.loganomaly import train_loganomaly

app = FastAPI()
# UPLOAD_DIR = "app/uploads"
templates = Jinja2Templates(directory="app/templates")
# os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/", response_class=HTMLResponse)
async def show_initial_form(request: Request):
    print("UPLOAD_DIR: ", UPLOAD_DIR)
    return templates.TemplateResponse("index.html", {"request": request, "result": ""})

@app.get("/uploadPage", response_class=HTMLResponse)
async def show_upload_form(request: Request):
    # print("UPLOAD_DIR: ", UPLOAD_DIR)
    return templates.TemplateResponse("upload.html", {"request": request, "result": ""})


@app.post("/trainUpload/")
async def handle_train_upload(request: Request, log_file: UploadFile = File(...), label_file: UploadFile = File(...)):
    print("UPLOAD_DIR: ", UPLOAD_DIR)
    log_path = os.path.join(UPLOAD_DIR, log_file.filename)
    label_path = os.path.join(UPLOAD_DIR, label_file.filename)

    with open(log_path, "wb") as f1:
        shutil.copyfileobj(log_file.file, f1)
    with open(label_path, "wb") as f2:
        shutil.copyfileobj(label_file.file, f2)

    return templates.TemplateResponse("uploadSuccess.html", {"request": request, "result": ""})

@app.post("/preprocess/")
async def do_preprocess(request: Request, logformat: str = Form(...)):
    
    return {"logformat": logformat}