from turtle import title
from urllib import request
import uvicorn
from fastapi import FastAPI, File, UploadFile
import cv2
from keras.models import Model
import tensorflow as tf
from keras.models import load_model
import numpy as np
from keras.applications import ResNet50
from keras.optimizers import Adam
from fastapi.templating import jinja2templates
from keras.models import Sequential, Model
from keras.utils import np_utils
from keras.preprocessing import image, sequence
import cv2
from keras_preprocessing.sequence import pad_sequences
from tqdm import tqdm
from app import *
from functions import *
from pickle import load,dump

#templates.Templateresponse("index.html",{"data":data})
app=FastAPI(title="Image Captioning")
templates=Jinja2templates(directory='/templates')


#Image Model

image_model = ResNet50(include_top=False,weights='imagenet',input_shape=(224, 224,3),pooling="avg")
new_input = image_model.input
hidden_layer = image_model.layers[-2].output

image_features_extract_model = tf.keras.Model(new_input, hidden_layer)


app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1


@app.route('/')
def index():
    return templates.render_template('index.html')

@app.route('/after', methods=['GET', 'POST'])
def after():

    global model, resnet, vocab, inv_vocab

    img = request.files['file1']

    img.save(r'static\file.jpg')

    
    image = cv2.imread(r'static\file.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224,224))
    image = np.reshape(image, (1,224,224,3))

    img_tensor_val=image_features_extract_model(image)
    start_token = tokenizer.word_index['<start>']
    end_token = tokenizer.word_index['<end>']

    #decoder input is start token.
    decoder_input = [start_token]
    output = tf.expand_dims(decoder_input, 0) #tokens
    result = [] #word list
    max_length=33
    for i in range(100):
        
        hidden = decoder.reset_state(batch_size=1)
        temp_input = tf.expand_dims(load_image(image)[0], 0)
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


