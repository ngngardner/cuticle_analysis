
from os import environ
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame

from .. import const


class Textbox():
    """
    This class handles textbox objects using the Pygame library.
    label is the default text shwon when value is empty.
    value is the value that user enters into the textbox.
    """

    def __init__(self, surface, font_size, default_label, default_value, size, position):
        self.surface = surface
        self.font_size = font_size
        self.font = pygame.font.SysFont('Arial', self.font_size)
        self.label = default_label
        self.value = default_value
        self.text_object = None
        self.size = size
        self.position = position
        self.function = None
        self.func_param = None
        self.text_color = const.TEXTBOX_LABEL_COLOR
        self.is_running = True
        self.shape_object = pygame.Rect(self.position[0], self.position[1], int(list(size.values())[list(
            size.keys()).index("width")]), int(list(size.values())[list(size.keys()).index("height")]))

    def __show__(self):
        # Sets textbox visible
        pygame.draw.rect(
            self.surface, const.TEXTBOX_BKG_COLOR, self.shape_object)
        if self.value is None:
            self.text_color = const.TEXTBOX_LABEL_COLOR
            self.text_object = self.font.render(
                self.label, True, self.text_color)
        else:
            self.text_color = (0, 0, 0)
            self.text_object = self.font.render(
                self.value, True, self.text_color)
        self.surface.blit(self.text_object, (self.position[0], (self.position[1] + int(
            list(self.size.values())[list(self.size.keys()).index("height")]/3))))
