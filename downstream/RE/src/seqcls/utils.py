import datetime
import json
import logging
import logging.config
import os
import random
import shutil
import time
from collections import defaultdict, deque
from pathlib import Path


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
