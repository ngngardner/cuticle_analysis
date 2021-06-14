
import logging
import os

import click
from click.termui import prompt
from rich import print

logger = logging.getLogger(__name__)


def default_dataset():
    download_dataset()
    unzip_dataset()


def download_dataset():
    import gdown

    print("Downloading dataset...")
    url = "https://drive.google.com/uc?id=1xABlWE790uWmT0mMxbDjn9KZlNUB6Y57"
    output = "./dataset.zip"
    gdown.download(url, output, quiet=False)

    assert os.path.isfile(output)
    print("Downloaded dataset!")


@click.command()
@click.option(
    "--path",
    prompt="Enter dataset.zip filepath. (default=./dataset.zip)",
    default=".",
    help="The path of the dataset.zip file to use.")
def unzip_dataset(path: str):
    print("Extracting dataset...")
    zip_path = f"{path}/dataset.zip"
    folder_path = f"./dataset"

    try:
        if not os.path.isdir(folder_path):
            if os.path.isfile(zip_path):
                from zipfile import PyZipFile
                zf = PyZipFile(zip_path)
                zf.extractall(path=folder_path)

                assert os.path.isdir(folder_path)
                print("Extracted dataset!")
            else:
                raise Exception(f"Failed to find {zip_path}.")
        else:
            raise Exception(f"Directory {folder_path} already exists.")

    except Exception as e:
        logger.error(f'Failed to unzip dataset:" {e}')
        raise e


menu = {
    "0": "Quit",
    "1": "Download dataset.zip from Google Drive",
    "2": "I already have dataset.zip"
}

actions = {
    "0": SystemExit,
    "1": default_dataset,
    "2": unzip_dataset
}


@click.command()
@click.option(
    "--dataset-init",
    type=click.Choice(list(menu.keys())),
    prompt="Failed to find dataset. How should dataset be initialized?")
def init_dataset(dataset_init: str):
    actions.get(dataset_init, None)()


def dataset_setup():
    if os.path.isdir("dataset"):
        print("Found dataset!")
        return
    else:
        for key in menu:
            print(f'{key}: {menu[key]}')
        init_dataset()

    assert(os.path.isdir("dataset"))
    print("Successfully initialized dataset.")
