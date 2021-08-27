# importing the required libaries
import os
import glob

import sys
sys.path.append(os.getcwd())

from azureml.core import Workspace
from azureml.core.webservice import LocalWebservice, Webservice

import numpy as np

from flask import Flask, render_template, request, redirect, url_for
import json

from PIL import Image
import requests

from utils import serialize_image, deserialize_mask
from image_display import blend_mask2image, add_legend2image


app = Flask(__name__)

@app.route('/')
def home():
    # Directories
    global src_dir, target_dir, static_dir
    static_dir = os.path.join(os.getcwd(),"static")
    files_dir = os.path.join(os.path.dirname(os.getcwd()),'files')
    # files_dir = os.path.join(os.path.dirname(os.getcwd()),'data','full')
    # files_dir = os.getcwd()
    src_dir = os.path.join(files_dir, "datasets", "src")
    target_dir = os.path.join(files_dir, "datasets", "target")
    
    return render_template('home.html', text=glob.glob(os.path.join(src_dir,'*'),recursive=True))


@app.route('/', methods=['POST', 'GET'])
def segment_image():
    if request.method == 'POST':

        # Delete existing image / predictions
        print(r'Delete existing image / predictions')
        files = glob.glob(os.path.join(static_dir,'*'))
        for f in files:
            os.remove(f)

        # Get the image name
        print('Get the image name')
        image_name = request.form['search']
        image_ext = os.path.splitext(image_name)[1]
        image_core = image_name[:-len(image_ext)]

        # Get the paths to the image and its mask
        print('Get the paths to the image and its mask')
        image_path = sorted(glob.glob(os.path.join(src_dir,'**',image_name + '*'), recursive=True))[0]
        mask_path = sorted(glob.glob(os.path.join(target_dir,'**',image_name[:-24] + '*'), recursive=True))[0]
        print(str(image_path))
        print(str(mask_path))

        # Load the image and its mask
        print('Load the image and its mask')
        img = Image.open(image_path)
        mask_true = np.array(Image.open(mask_path))[:,:,0]

        # Send the prediction request to the server
        print('Send the prediction request to the server')
        headers = {'Content-Type': 'application/json'}
        scoring_uri = r"http://b9296d37-0940-4281-a63d-8353dfd4b401.westeurope.azurecontainer.io/score"
        resp = requests.post(scoring_uri, data=serialize_image(img), headers=headers)

        # Process the prediction made by the server
        print('Process the prediction made by the server')
        mask_predict = deserialize_mask(json.loads(resp.json())['mask'])
        cat2color = json.loads(resp.json())['code']
        cat2color = {int(cat): (tuple(cat2color[cat][0]), cat2color[cat][1]) for cat in cat2color}
        blend_predict = blend_mask2image(img, mask_predict, (300,500), cat2color)
        blend_predict = add_legend2image(blend_predict, cat2color, 'right')
        blend_predict.save(os.path.join(static_dir, image_core + '_predict' + image_ext))

        # Process the true mask
        print('Process the true mask')
        blend_true = blend_mask2image(img, mask_true, (300,500), cat2color)
        # blend_true = add_legend2image(blend_true, cat2color)
        blend_true.save(os.path.join(static_dir, image_core + '_true' + image_ext))

        # Save the original image in the static folder
        print('Save the original image in the static folder')
        img.save(os.path.join(static_dir, image_name))

    return render_template('home.html',
                            orig=url_for('static', filename=image_name),
                            blend_true=url_for('static', filename=image_core + '_true' + image_ext),
                            blend_predict=url_for('static', filename=image_core + '_predict' + image_ext))