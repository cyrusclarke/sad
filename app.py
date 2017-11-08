import os
# from flask import Flask, render_template, request
#ACTIVATE YOUR CONDA INSTANCE FOR THIS TO WORK!
import numpy as np
import base64
from flask import Flask, abort, jsonify, render_template, request, make_response, send_from_directory
import cPickle as pickle 
# from sklearn.externals import joblib
from io import BytesIO
import vgg16
from vgg16 import Vgg16
from skimage import io as skio
from skimage.transform import resize

# imports from jupyter
import json
np.set_printoptions(precision=4, linewidth=100)
from matplotlib import pyplot as plt
import utils; reload(utils)
from utils import plots
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from matplotlib.pyplot import imshow

template_dir = os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

app = Flask(__name__, template_folder="/")	

APP_ROOT=os.path.dirname(os.path.abspath(__file__))

my_model = pickle.load(open("data.pkl", "rb"))


#home route
@app.route("/")
def index():
    return render_template("index.html")

#upload route
#start by specifying your route and method
@app.route("/index", methods=['POST'])
#then create the upload function
def upload():
	#set target equal to the location of our images
	target = os.path.join(APP_ROOT, 'images/')
	print(target)

	if not os.path.isdir(target):
		os.mkdir(target)

	for file in request.files.getlist("file"):
		print(file)
		filename = file.filename
		destination = "/".join([target, filename])
		print(destination)
		file.save(destination)

	return render_template("index.html", image_name=filename)
#this display is wreaking havoc with my paths!
@app.route('/display/<filename>')
def send_image(filename):
	return send_from_directory('images', filename)


#machine learning script
@app.route('/api', methods=['POST'])
def make_predict():
	#get JSON from the post //is this 'data'?
	# data = request.get_json(silent=True)['image']
	data = request.get_json(force=True)
	#Take the url that is coming through from JSON and split it so you get just the FILE
	print("THE DATA = "+data.split("/")[-1])
	filename = data.split("/")[-1]
	#store the filename (absolute path) in a variable
	img = skio.imread("./images/"+filename)
	print("THIS WORKS",img)
	#resize the image to the expected format
	# img = resize(img, (224,224))
	# you can use this to test the data returns through here --> return make_response(str(data))
	#run the model preduct function on the image. might need to reshape array
#FUNCTIONING CODE HERE
	img = resize(img, (224,224))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	predictions = my_model.predict(x, True)
	return make_response(str(predictions))

	#run through model
	# number = my_model.predict(img.reshape(1,-1)[0])
	# return make_response(str(number)) 



	#convert our JSON to a numpy array
	# data = data[22:]
	# return make_response(str(data))
	# # print( "C O N V E R T E D  TO  N U M P Y ="+data)
	# print( "N O W  R E A D I N G = "+img)
	# return make_response(str(img))

	# img = skio.imread(BytesIO(base64.b64decode(data)))[:,:,3]
	# img = resize(img, (224,224))
	# # number = my_model.predict(data.reshape(1,-1)[0])
	# print(number)
	# return make_response(str(number))



if __name__ == '__main__':
    app.run()