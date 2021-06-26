
import pygame  # GUI Framework
from . import const


class SpeciesPane():
    def __init__(self, surface, position, data):
        self.surface = surface
        self.position = position
        self.data = data
        self.size = (200, 800)
