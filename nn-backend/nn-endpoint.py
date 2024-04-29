import io
import flask
import cairosvg
import numpy as np
from flask_cors import CORS
from PIL import Image
from nn import MLP
from helper import shift_vector, trim_image
from pathlib import Path
import matplotlib.image
import cv2
from scipy.ndimage import zoom

app = flask.Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

path = Path(__file__).parent / "data"

MODEL_PARAMS = [784, 128, 64, 64, 10]
mnist = MLP(MODEL_PARAMS, "data/weights.npy", "data/biases.npy", load_existing=True)

@app.route("/api/", methods=['GET'])
def index():
    response = flask.jsonify({"endpoints": ["/api/query_mnist/"]})
    return response

@app.route("/api/query_mnist", methods=['POST'])
def query_mnist():
    mnist_svg = flask.request.get_json()["mnist_svg"]
    mem = io.BytesIO()
    cairosvg.svg2png(mnist_svg, output_width=28, output_height=28, write_to=mem, negate_colors=True)
    cairosvg.svg2png(mnist_svg, output_width=28, output_height=28, write_to="inital_image.png", negate_colors=True)

    x = np.array(Image.open(mem))

    # Only consider first three channels (r,g,b), then convert to grayscale
    x = x[:,:,:3]
    x = x.mean(axis=2)

    # Calculate center of mass to center images
    # center_x = np.argmax(x.mean(axis=1))
    # center_y = np.argmax(x.mean(axis=0))
    # x = shift_vector(x, center_x, center_y, 28, 28)
    # matplotlib.image.imsave('before_flattening.png', x)
    # Flatten image, normalize
    x = trim_image(x, 28, 28)
    x = x.flatten().reshape(784, 1)
    x = x / 256.0
    # print(x)
    # matplotlib.image.imsave('after_flattening.png', x)
    
    new_image = Image.open('trimmed_rows_and_cols.png')
    print(new_image.size)
    new_image = new_image.resize((24, 24), Image.Resampling.LANCZOS)
    new_x = np.array(new_image)
    new_x = new_x[:,:,:3]
    new_x = new_x.mean(axis=2)
    num_to_add = 2
    new_x = np.insert(new_x, 0, np.zeros((num_to_add, new_x.shape[1])), axis=0)
    new_x = np.append(new_x, np.zeros((num_to_add, new_x.shape[1])), axis=0)  
    new_x = new_x.T  
    new_x = np.insert(new_x, 0, np.zeros((num_to_add, new_x.shape[1])), axis=0)
    new_x = np.append(new_x, np.zeros((num_to_add, new_x.shape[1])), axis=0)  
    new_x = new_x.T
    matplotlib.image.imsave('trimmed_and_resized.png', new_x)    
    new_x[new_x < 80.0] = 0
    matplotlib.image.imsave('hopefully_fixed.png', new_x)        
    new_x = new_x.flatten().reshape(784, 1)
    new_x = new_x / 256.0


    # Generate prediction
    # y_pred, _, _ = mnist.forward(x)
    y_pred, _, _ = mnist.forward(x)
    # print(y_pred)
    response = flask.jsonify({"0": str(y_pred[0][0]),
                              "1": str(y_pred[1][0]),
                              "2": str(y_pred[2][0]),
                              "3": str(y_pred[3][0]),
                              "4": str(y_pred[4][0]),
                              "5": str(y_pred[5][0]),
                              "6": str(y_pred[6][0]),
                              "7": str(y_pred[7][0]),
                              "8": str(y_pred[8][0]),
                              "9": str(y_pred[9][0])})
    return response