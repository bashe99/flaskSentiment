import sys
import os
from model.data_utils import CoNLLDataset, get_CoNLL_dataset
from model.ner_model import NERModel
from model.embedding_projection_ner_model import ProjectionNERModel
from model.config import Config
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import glob
import pickle

from flask import Flask
app = Flask(__name__)
globalModel = None

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/user/<username>')
def show_user_profile(username):
    # show the user profile for that user
    return 'User %s' % username

@app.route('/build')
def buildModel():
    global globalModel
    config = Config()
    # build model
    if config.use_embedding_proj_pred:
        globalModel = ProjectionNERModel(config)
    else:
        globalModel = NERModel(config)
    globalModel.build()
    globalModel.restore_latest_session(config.dir_model_evaluate)
    return 'Success'

@app.route('/detect/<sentence>')
def detect(sentence):
    global globalModel
    pred_domain, pred_intent, pred_tags = globalModel.predict(sentence, 0)
    return pred_intent

if __name__ == "__main__":
    app.run()
    buildModel()