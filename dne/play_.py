import os
os.environ["TORCH_HOME"] = "/disks/sdb/torch_home"

from awesome_glue.config import Config
config = Config()._parse_args()
if not config.alchemist:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config.cuda)

import logging
from awesome_glue.task import Task
from allennlp.common import logging as common_logging
from luna.ram import ram_write
from luna.logging import log, log_config
from luna import set_seed
import sys

logging.getLogger().setLevel(logging.WARNING)
log_config("log", "c")
log(config)
sys.stdout.flush()
set_seed(config.seed)
if config.alchemist:
    common_logging.FILE_FRIENDLY_LOGGING = True

ram_write("config", config)
from flask import Flask,request

import copy
cnn_imdb_config = copy.deepcopy(config)
cnn_imdb_config.task_id="IMDB"
cnn_imdb_config.arch = "cnn"
cnn_imdb_task = Task(cnn_imdb_config)
print("="*10+"cnn_imdb"+"="*10)
cnn_imdb_task.evaluate_predictor()


cnn_ag_config = copy.deepcopy(config)
cnn_ag_config.task_id = "AGNEWS"
cnn_ag_config.arch = "cnn"
cnn_ag_task = Task(cnn_ag_config)
print("="*10+"cnn_ag"+"="*10)
cnn_ag_task.evaluate_predictor()


lstm_imdb_config = copy.deepcopy(config)
lstm_imdb_config.task_id="IMDB"
lstm_imdb_config.arch = "lstm"
lstm_imdb_task = Task(lstm_imdb_config)
print("="*10+"lstm_imdb"+"="*10)
lstm_imdb_task.evaluate_predictor()

lstm_ag_config = copy.deepcopy(config)
lstm_ag_config.task_id = "AGNEWS"
lstm_ag_config.arch = "lstm"
lstm_ag_task = Task(lstm_ag_config)
print("="*10+"lstm_ag"+"="*10)
lstm_ag_task.evaluate_predictor()

# Speed up the predictor
if cnn_imdb_config.arch != 'bert':
    cnn_imdb_task.predictor.set_max_tokens(360000)
    if cnn_imdb_config.nbr_2nd[1] == '2':
        if cnn_imdb_config.nbr_num <= 12:
            cnn_imdb_task.predictor.set_max_tokens(360000)
        elif cnn_imdb_task.config.nbr_num <= 24:
            cnn_imdb_task.predictor.set_max_tokens(120000)
        else:
            cnn_imdb_task.predictor.set_max_tokens(90000)
else:
    cnn_imdb_task.predictor.set_max_tokens(60000)

# Speed up the predictor
if cnn_ag_config.arch != 'bert':
    cnn_ag_task.predictor.set_max_tokens(360000)
    if cnn_ag_config.nbr_2nd[1] == '2':
        if cnn_ag_config.nbr_num <= 12:
            cnn_ag_task.predictor.set_max_tokens(360000)
        elif cnn_ag_task.config.nbr_num <= 24:
            cnn_ag_task.predictor.set_max_tokens(120000)
        else:
            cnn_ag_task.predictor.set_max_tokens(90000)
else:
    cnn_ag_task.predictor.set_max_tokens(60000)

if lstm_imdb_config.arch != 'bert':
    lstm_imdb_task.predictor.set_max_tokens(360000)
    if lstm_imdb_config.nbr_2nd[1] == '2':
        if lstm_imdb_config.nbr_num <= 12:
            lstm_imdb_task.predictor.set_max_tokens(360000)
        elif lstm_imdb_task.config.nbr_num <= 24:
            lstm_imdb_task.predictor.set_max_tokens(120000)
        else:
            lstm_imdb_task.predictor.set_max_tokens(90000)
else:
    lstm_imdb_task.predictor.set_max_tokens(60000)

# Speed up the predictor
if lstm_ag_config.arch != 'bert':
    lstm_ag_task.predictor.set_max_tokens(360000)
    if lstm_ag_config.nbr_2nd[1] == '2':
        if lstm_ag_config.nbr_num <= 12:
            lstm_ag_task.predictor.set_max_tokens(360000)
        elif lstm_ag_task.config.nbr_num <= 24:
            lstm_ag_task.predictor.set_max_tokens(120000)
        else:
            lstm_ag_task.predictor.set_max_tokens(90000)
else:
    lstm_ag_task.predictor.set_max_tokens(60000)

import json
import numpy as np
app = Flask(__name__)
@app.route("/cnn_imdb",methods=['GET','POST'])
def cnn_imdb():
    data = request.data

    data = data.decode("utf-8")
    rel = {"sent":data}

    result = cnn_imdb_task.predictor.predict_json(rel)
    result = result['probs']
    label = np.argmax(result)
    label = str(label)
    return {"result":label}

@app.route("/lstm_imdb",methods=['GET','POST'])
def lstm_imdb():
    data = request.data
    data = data.decode("utf-8")
    rel = {"sent": data}
    result = lstm_imdb_task.predictor.predict_json(rel)
    result = result['probs']
    label = np.argmax(result)
    label = str(label)
    return {"result": label}

@app.route("/cnn_ag",methods=['GET','POST'])
def cnn_ag():
    data = request.data
    data = data.decode("utf-8")
    rel = {"sent": data}
    result = cnn_ag_task.predictor.predict_json(rel)
    result = result['probs']
    label = np.argmax(result)
    label = str(label)
    return {"result": label}

@app.route("/lstm_ag",methods=['GET','POST'])
def lstm_ag():
    data = request.data
    data = data.decode("utf-8")
    rel = {"sent": data}
    result = lstm_ag_task.predictor.predict_json(rel)
    result = result['probs']
    label = np.argmax(result)
    label = str(label)
    return {"result": label}

app.run(host='0.0.0.0',port='8999')

