
from os import environ
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'  # noqa
import pygame

import threading

import click
from pygame.locals import HIDDEN, DOUBLEBUF

from cuticle_analysis.datasets import Dataset
from .objects.image_viewer import ImageViewer
from .objects.buttons.buttons import Buttons
from .objects.textbox import Textbox
from . import const


@click.command()
def application():
    # Temporary check verifying that the window staus is open until user quits.
    gui_thread = threading.Thread(target=start)
    pygame.init()
    gui_thread.start()
    gui_thread.join()


class Gui:
    # Object constructor
    def __init__(self, surface):
        self.surface = surface
        self.increment_id = False
        self.decrement_id = False

    # Overloaded function from the pygame library that sets caption from surface name which
    # is concatenated with PROGRAM_NAME
    def set_caption(self, surface_name) -> str:
        caption = surface_name + " - " + const.PROGAM_NAME
        pygame.display.set_caption(caption)

    # Returns the surface object.
    def get_surface(self) -> pygame.display:
        return self.surface

    # Sets the surface object.
    def set_surface(self, surface):
        self.surface = surface


def start():
    """
    This file contains the gui functions for the main window.
    The function opens the main window and adds the imageviewer navigation buttons with their corresponding
    event listeners, image id, image, ant's species classification, and the ant's texture classification.
    """
    data = Dataset(size=(16, 16), dataset_type='rough_smooth')
    # Initializes and launches window.
    window = pygame.display.set_mode(
        (const.WINDOW_SIZE[0], const.WINDOW_SIZE[1]), HIDDEN)
    main = Gui(window)
    white = (255, 255, 255)
    __image_id__ = 1
    main.set_caption(str(__image_id__) + ".jpg")
    main.get_surface().fill(white)
    pygame.mouse.set_visible(1)
    body_font = pygame.font.SysFont('Arial', const.BODY_FONT_SIZE)
    width = main.get_surface().get_width()
    height = main.get_surface().get_height()
    standby_text_pos = (width/2, height/2)
    standby_text = body_font.render("Loading program...", True, (0, 0, 0))
    main.get_surface().blit(
        standby_text, (standby_text_pos[0], standby_text_pos[1]))
    pygame.display.set_mode(
        (const.WINDOW_SIZE[0], const.WINDOW_SIZE[1]), DOUBLEBUF)
    pygame.display.update()
    prev_bttn_pos = (225, 475)
    next_bttn_pos = (525, 475)
    id_text_pos = ((width/3) + 25, 50)
    id_txtbox_pos = (300, 475)
    id_textbox = Textbox(main.get_surface(), 16, "Enter ID to view image...", None, {
                         "width": 200, "height": 50}, [id_txtbox_pos[0], id_txtbox_pos[1], 50])
    previous_button = Buttons(main.get_surface(), "rectangle", "<", const.BUTTON_COLOR, {
                              "width": 50, "height": 50}, [prev_bttn_pos[0], prev_bttn_pos[1], 50])
    next_button = Buttons(main.get_surface(), "rectangle", ">", const.BUTTON_COLOR, {
                          "width": 50, "height": 50}, [next_bttn_pos[0], next_bttn_pos[1], 50])
    id_text = body_font.render(str(__image_id__), True, (0, 0, 0))
    ant_iv = ImageViewer(main.get_surface(), data,
                         'rough_smooth', (225, 100), (350, 350))
    next_bttn_listener = threading.Thread(target=next_button.on_click, args=[
                                          lambda: ant_iv.__set_increment_flag__(True)])
    next_bttn_listener.start()
    prev_bttn_listener = threading.Thread(target=previous_button.on_click, args=[
                                          lambda: ant_iv.__set_decrement_flag__(True)])
    prev_bttn_listener.start()
    ant_iv.__show__()
    previous_button.show()
    next_button.show()
    id_textbox.__show__()
    main.get_surface().blit(id_text, [id_text_pos[0], id_text_pos[1]])
    pygame.display.update()
    is_running = True
    while is_running == True:
        main.set_caption(str(__image_id__) + ".jpg")
        main.get_surface().fill(white)
        previous_button = Buttons(main.get_surface(), "rectangle", "<", (200, 200, 200), {
                                  "width": 50, "height": 50}, [prev_bttn_pos[0], prev_bttn_pos[1], 50])
        next_button = Buttons(main.get_surface(), "rectangle", ">", (200, 200, 200), {
                              "width": 50, "height": 50}, [next_bttn_pos[0], next_bttn_pos[1], 50])
        id_text = body_font.render(
            "Image ID: " + str(__image_id__), True, (0, 0, 0))

        main.get_surface().blit(id_text, [id_text_pos[0], id_text_pos[1]])
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                is_running = False
                previous_button.set_running_status(is_running)
        if ant_iv.__get_increment_flag__() == True:
            if __image_id__ < 1773:
                __image_id__ += 1
        if ant_iv.__get_decrement_flag__() == True:
            if __image_id__ > 1:
                __image_id__ -= 1
                ant_iv.__update_image__(__image_id__)
                ant_iv.__show__()
                ant_iv.__set_decrement_flag__(False)
        ant_iv.__update_image__(__image_id__)
        ant_iv.__show__()
        previous_button.show()
        next_button.show()
        id_textbox.__show__()
        ant_iv.__set_increment_flag__(False)
        ant_iv.__set_decrement_flag__(False)
        pygame.display.update()
    next_bttn_listener.join()
    prev_bttn_listener.join()
