
from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import shutil, os,sys
from .config import UPLOAD_DIR, TRAIN_DIR
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from deeplog import train
# from data_process_pred import 
# from models.loganomaly import train_loganomaly
from uiUtilities import ui_predict, ui_train
app = FastAPI()
# UPLOAD_DIR = "app/uploads"
templates = Jinja2Templates(directory="log_anomaly_ui/app/templates")
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
    print("handle_train_upload UPLOAD_DIR: ", UPLOAD_DIR)
    log_path = os.path.join(UPLOAD_DIR, log_file.filename)
    label_path = os.path.join(UPLOAD_DIR, label_file.filename)

    with open(log_path, "wb") as f1:
        shutil.copyfileobj(log_file.file, f1)
    with open(label_path, "wb") as f2:
        shutil.copyfileobj(label_file.file, f2)

    return templates.TemplateResponse("uploadSuccess.html", {"request": request, "result": ""})

@app.post("/preprocess/")
async def do_preprocess(request: Request, logformat: str = Form(...)):
    print("in do_preprocess:", UPLOAD_DIR)
    preProcResult = ui_train(input_dir=UPLOAD_DIR, output_dir=TRAIN_DIR)
    return {preProcResult}