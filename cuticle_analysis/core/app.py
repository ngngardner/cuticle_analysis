
import logging

import click
from rich import print

from .dataset import dataset_setup

logger = logging.getLogger(__name__)


@click.command()
@click.pass_context
def start(ctx):
    print("Welcome to Ant Cuticle Analysis.")
    ctx.invoke(dataset_setup)
