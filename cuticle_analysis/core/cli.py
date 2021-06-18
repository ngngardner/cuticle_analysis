
import click

from .app import start
from .dataset import dataset_setup, unzip_dataset, download_dataset, download_unzip


@click.group()
def cli():
    pass


cli.add_command(start)
cli.add_command(dataset_setup)
cli.add_command(download_dataset)
cli.add_command(unzip_dataset)
cli.add_command(download_unzip)
