import os
# from flask import Flask, render_template, request

import numpy as np
import base64
from flask import Flask, abort, jsonify, render_template, request, make_response, send_from_directory
import cPickle as pickle 
from io import BytesIO
import vgg16
from vgg16 import Vgg16
from skimage import io as skio
from skimage.transform import resize


app = Flask(__name__)

APP_ROOT=os.path.dirname(os.path.abspath(__file__))

# my_model = pickle.load(open("data.pkl", "rb"))


#home route
@app.route("/")
def index():
    return render_template("upload.html")

#upload route
#start by specifying your route and method
@app.route("/upload", methods=['POST'])
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

	return render_template("upload.html", image_name=filename)

@app.route('/display/<filename>')
def send_image(filename):
	return send_from_directory('images', filename)


#machine learning script
@app.route('/api', methods=['POST'])
def make_predict():
	#get JSON from the post //is this 'data'?
	data = request.get_json(silent=True)['image']
	print("THE DATA = "+data)
	return make_response(str(data))
	#convert our JSON to a numpy array
	# data = data[22:]
	# print( "C O N V E R T E D  TO  N U M P Y ="+data)
	# img = skio.imread(data)[:,:,3]
	# print( "N O W  R E A D I N G = "img)
	# return make_response(str(img))

	# img = skio.imread(BytesIO(base64.b64decode(data)))[:,:,3]
	# img = resize(img, (224,224))
	# # number = my_model.predict(data.reshape(1,-1)[0])
	# print(number)
	# return make_response(str(number))



if __name__ == '__main__':
    app.run()