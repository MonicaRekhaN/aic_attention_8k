import uvicorn
import cv2
import tensorflow as tf
from keras.models import load_model
import numpy as np
from keras.models import Model
import cv2
from functions import *
from pickle import load
from keras.applications.resnet import ResNet50
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi import FastAPI,Request,UploadFile,File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.datastructures import URL
from PIL import Image
import io
import base64
import numpy as np

#templates.Templateresponse("index.html",{"data":data})
app = FastAPI(Title="Image Captioning")
templates = Jinja2Templates(directory="templates")
templates.env.globals['URL'] = URL
# app.mount('/', StaticFiles(directory="static",html=True),name="static")
app.mount("/static", StaticFiles(directory="static"), name="static")

image_model = ResNet50(include_top=False,weights='imagenet',input_shape=(224, 224,3),pooling="avg")
new_input = image_model.input
hidden_layer = image_model.layers[-2].output

image_features_extract_model = tf.keras.Model(new_input, hidden_layer)


@app.get("/",response_class=HTMLResponse)
def form_get(request: Request):
    print("testing")
    return templates.TemplateResponse('form.html', context={'request': request})

@app.post("/predict",response_class=HTMLResponse)
async def form_post(request: Request,file: UploadFile = File(...)):

    file_contents = file.file.read()
    file_location = "static/file.jpg"
    with open(file_location, "wb+") as file_object:
        file_object.write(file_contents)
    
    npimg = np.fromstring(file_contents,np.uint8)
    img = cv2.imdecode(npimg,cv2.IMREAD_COLOR)
    img=cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img.astype("uint8"))
    rawBytes = io.BytesIO()
    img.save(rawBytes, "JPEG")
    rawBytes.seek(0)
    img_base64 = base64.b64encode(rawBytes.getvalue()).decode('ascii')
    mime = "image/jpeg"
    uri = "data:%s;base64,%s"%(mime, img_base64)

    start_token = tokenizer.word_index['<start>']
    end_token = tokenizer.word_index['<end>']

    #decoder input is start token.
    decoder_input = [start_token]
    output = tf.expand_dims(decoder_input, 0) #tokens
    result = [] #word list
    max_length=33
    for i in range(100):
        
        hidden = decoder.reset_state(batch_size=1)
        temp_input = tf.expand_dims(load_image(file_location)[0], 0)
        img_tensor_val = image_features_extract_model(temp_input)
        img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))
        features = encoder(img_tensor_val)
        dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
        result = []

        for i in range(max_length):
            predictions, hidden, attention_weights = decoder(dec_input, features, hidden)
            predicted_id = tf.argmax(predictions[0]).numpy()
            result.append(tokenizer.index_word[predicted_id])

            if tokenizer.index_word[predicted_id] == '<end>':
                return result

            dec_input = tf.expand_dims([predicted_id], 0)

   

    return templates.render_template('after.html', data=result)

# if __name__ == "__main__":
#     app.run(debug=True)
if __name__ == "__main__":
    initialize()
    uvicorn.run(app, debug=True)


