from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.routing import APIRouter
import onnxruntime
import numpy as np
from PIL import Image
from torchvision import transforms
from starlette.responses import FileResponse
import shutil
import os

#Creating FastAPI app instance
app = FastAPI(title="Cat vs Dog Classification API")

# Creating routers for Scratch and Pretrained models
scratch_router = APIRouter()
pretrained_router = APIRouter()

# Loading the ONNX models for both Scratch and Pretrained
scratch_ort_session = onnxruntime.InferenceSession('scratch_model.onnx' )
pretrained_ort_session = onnxruntime.InferenceSession('pretrained_model.onnx')

# Defining preprocessing transform for images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Defining the index and version endpoints
@app.get("/")
async def index():
    return {"message": "Cat vs Dog Image Classification API (Rifat)"}

@app.get("/version")
async def version():
    return {"version": "1.0"}

# Defining routes for the Scratch model
@scratch_router.post("/v1/predict/")
async def predict_scratch(file: UploadFile):
    try:
        # Save the uploaded image temporarily
        with open("temp_image.jpg", "wb") as temp_image:
            shutil.copyfileobj(file.file, temp_image)
        
        # Preprocess the input image
        input_image = Image.open("temp_image.jpg")
        input_image = transform(input_image)
        input_image = np.array(input_image).reshape(1, 3, 224, 224).astype(np.float32)

        # To run inference with the Scratch model
        ort_inputs = {scratch_ort_session.get_inputs()[0].name: input_image}
        ort_outputs = scratch_ort_session.run(None, ort_inputs)

        # To return the model's output
        predicted_class = np.argmax(ort_outputs[0])
        class_labels = ["cat", "dog"]
        predicted_label = class_labels[predicted_class]

        # Removing temporary image file
        os.remove("temp_image.jpg")

        return {"prediction": predicted_label}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Defining routes for the Pretrained model
@pretrained_router.post("/v1/predict/")
async def predict_pretrained(file: UploadFile):
    try:
        # Save the uploaded image temporarily
        with open("temp_image.jpg", "wb") as temp_image:
            shutil.copyfileobj(file.file, temp_image)
        
        # Preprocessing the input image
        input_image = Image.open("temp_image.jpg")
        input_image = transform(input_image)
        input_image = np.array(input_image).reshape(1, 3, 224, 224).astype(np.float32)

        # To run inference with the Pretrained model
        ort_inputs = {pretrained_ort_session.get_inputs()[0].name: input_image}
        ort_outputs = pretrained_ort_session.run(None, ort_inputs)

        # Returning the model's output
        predicted_class = np.argmax(ort_outputs[0])
        class_labels = ["cat", "dog"]
        predicted_label = class_labels[predicted_class]

        # Removing temporary image file
        os.remove("temp_image.jpg")

        return {"prediction": predicted_label}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Mounting routers for Scratch and Pretrained models under their respective paths
app.include_router(scratch_router, prefix="/scratch")
app.include_router(pretrained_router, prefix="/pretrained")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
    
#to run in terminal
# uvicorn filename:object --reload (uvicorn app:app --reload)