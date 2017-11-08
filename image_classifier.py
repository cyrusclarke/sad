import numpy as np
import base64
from flask import Flask, abort, jsonify, render_template, request, make_response
import cPickle as pickle 
from io import BytesIO
import vgg16
from vgg16 import Vgg16
from skimage import io as skio
from skimage.transform import resize

app = Flask(__name__)

my_model = pickle.load(open("data.pkl", "rb"))

#upload image template
@app.route("/")
def index():
    return render_template("upload.html")

#call api
@app.route('/api', methods=['POST'])
def make_predict():
	#get JSON from the post
	data = request.get_json(silent=True)['image']
	console.log(data)
	#convert our JSON to a numpy array
	# data = data[22:]
	img = skio.imread(BytesIO(base64.b64decode(data)))[:,:,3]
	img = resize(img, (224,224))
	number = my_model.predict(x, True)
	print(number)
	return make_response(str(number))


	# # predict_request = data['v1']
	# #put the data into a np array
	# # predict_request = np.array[predict_request]
	# #run the np array through the pickled data
	# y_hat = my_data.predict(predict_request)
	# #return predicion (only 1)
	# output = [y_hat[0]]
	# #take the list and convert to JSON
	# return jsonify(results=output)



if __name__ == '__main__':
    app.run()