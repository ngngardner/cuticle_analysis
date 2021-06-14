
import logging
import os

import click
from rich import print

from .dataset import dataset_setup


logger = logging.getLogger(__name__)


class App():
    def __init__(self):
        pass

    def start(self):
        print("Welcome to Ant Cuticle Analysis.")

        dataset_setup()
