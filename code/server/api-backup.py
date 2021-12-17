from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from generate import truncate_noise, sample_image, load_checkpoint
from modelCopy import StyleGenerator
# from fastapi.responses import StreamingResponse
from io import BytesIO
import cv2
import torch
from torchvision.utils import save_image
from pathlib import Path
import requests
import os
from lipgan_evaluation import get_vid
import shutil
import aiofiles

API_KEY = "ae05a152c7f347b1b05975c321a0bbd2"


# root = Path("C:/Users/Admin/Desktop/btp/app/public/fakefaces/")
# for path in root.iterdir():
#     if path.is_file():
#         path.unlink()



app = FastAPI(title="API Endpoints for WARP")
origins = [
    "http://localhost",
    "http://127.0.0.1:8000",
    "http://localhost:4200",
    "http://localhost:4000",
    "http://localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
generator = StyleGenerator()
load_checkpoint("generator.pth", generator)


@app.get("/")
def root():
    return {"title": "endpoints for WARP"}


@app.get("/get-image/")
def get_image():
    """
    ## Endpoint to sample an image from the StyleGAN

    The stylegan is pre-loaded, the checkpoint file can be changed.  
    A `get` request to this endpoint returns the image as a streaming response.  
    """
    noise = truncate_noise(1, 512, 0.8)  # Get the noise -> 1 tensor only
    # noise = torch.zeros(1, 512)
    # Get the output from the generator, convert to numpy array
    image = sample_image(generator, noise) #.permute(1, 2, 0) #.numpy() 
    # image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)  # Convert from float32 to uint8
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # From BGR to RGB
    root = Path("C:/Users/Admin/Desktop/btp/app/public/fakefaces/")
    image_paths = [path for path in root.iterdir() if path.is_file()]
    curr_count = len(image_paths)
    new_path = root / f"fakeface{curr_count + 1}.png"
    save_image(image, str(new_path))
    # cv2.imwrite(str(new_path), image)
    # _, image = cv2.imencode(".png", image)  # Encode the image into a standard image format

    return {"path": f"fakeface{curr_count + 1}.png"}
    
@app.post("/get-video/")
def get_video(voice: str, text:str, image_name:str):
    """
    ## Endpoint to generate a video from the lipGAN model

    """
    res = requests.get(f'http://api.voicerss.org/?key={API_KEY}&hl=en-us&c=WAV&src={text}&v={voice}')
    root = Path("C:/Users/Admin/Desktop/btp/app/public/voices/")
    voice_paths = [path for path in root.iterdir() if path.is_file()]
    curr_count = len(voice_paths)
    newpath = root / f"voice{curr_count+1}.wav"
    with open(str(newpath), 'wb') as fp:
        fp.write(res.content)

    img_file = f'C:/Users/Admin/Desktop/btp/app/public/fakefaces/{image_name}'
    aud_file = str(newpath)
    root = Path("C:/Users/Admin/Desktop/btp/app/public/vid/")
    video_paths = [path for path in root.iterdir() if path.is_file()]
    curr_count1 = len(video_paths)
    outputfile = f"video{curr_count1+1}"
    get_vid(img_file, aud_file, outputfile, curr_count1+1)
    return {"path": f'{outputfile}.mp4'}

@app.post("/post-upload-face/")
async def uplaod_face(image :UploadFile=File(...)):
    """
        ## Endpoint for uploading image
    """
    root = Path("C:/Users/Admin/Desktop/btp/app/public/fakefaces/")
    image_paths = [path for path in root.iterdir() if path.is_file()]
    curr_count = len(image_paths)
    new_path = root / f"uploadface{curr_count + 1}.png"
    async with aiofiles.open(str(new_path), "wb") as outimage:
        content = await image.read()
        await outimage.write(content)

    return {"path": f"uploadface{curr_count + 1}.png"}




