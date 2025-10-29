import os
import sys


vlmevalkit_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'VLMEvalKit')
sys.path.append(vlmevalkit_path)

from vlmeval.config import supported_VLM

import argparse
import torch
import json
from tqdm import tqdm
import shortuuid
import copy
import hashlib
from PIL import Image
import logging
from constants import *
import time
logger = logging.getLogger(__name__)


import re

logger = logging.getLogger(__name__)


class VLMEvalModel:
    def __init__(self, model_type,max_new_token):
        model_name = model_type.split('vlmevalkit_')[-1]
        self.model = supported_VLM[model_name](max_new_tokens=max_new_token)


    def infer(self, messages,dataset_type):
        response = None
        while response is None:
            try:
                response = self.model.generate(message = messages, dataset=dataset_type) # the dataset here is only for not throwing an error

                return (True,response)

            except Exception as e:
                print(f"Error occurred: {e}")
                print('Retrying...')
                time.sleep(1)  
                continue

