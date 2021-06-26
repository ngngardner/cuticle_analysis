
from os import environ
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'  # noqa
import pygame

import threading


class Buttons():
    # Object constructor.
    # Pre-condition: surface must be a pygame.Surface object.
    def __init__(self, surface, shape_name, text_string, color, size, position):
        self.surface = surface
        self.text_string = text_string
        self.font = pygame.font.SysFont('Arial', int(
            list(size.values())[list(size.keys()).index("width")]))
        self.text_object = self.font.render(self.text_string, True, (0, 0, 0))
        self.color = color
        self.size = size
        self.position = position
        self.shape_name = shape_name
        self.shape_object = None
        self.func = None
        self.func_param = None
        self.is_running = True
        self.mouse_pos = pygame.mouse.get_pos()
        self.__create_button__()

    def __create_button__(self):
        "Calls shape constructor from pygame library based on provided shape from shape variable."
        cases = {
            "rectangle": self.__create_rectangle(),
            # "circle": __create_circle(self),
        }
        cases.get(self.shape_name,
                  "The self.shape_name is empty, so the button failed to create.")

    def __create_rectangle(self):
        "Creates a rectangle object."
        self.shape_object = pygame.Rect(
            self.position[0], self.position[1], self.get_width(), self.get_height())

    def show(self):
        "Sets button visible."
        cases = {
            "rectangle": pygame.draw.rect(self.surface, self.color, self.shape_object),
        }
        cases.get(self.shape_name,
                  "The self.shape_name is empty, so the button could not be shown.")
        self.surface.blit(self.text_object, ((
            self.position[0] + self.get_width()/4), (self.position[1])))

    def start_click_listener(self):
        "Runs loop that handles mouse click on button object."
        while self.is_running == True:
            self.mouse_pos = pygame.mouse.get_pos()
            for event in pygame.event.get():
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if (self.position[0] <= self.mouse_pos[0] <= self.position[0]+self.get_width()
                            and self.position[1] <= self.mouse_pos[1] <= self.position[1]+self.get_height()):
                        self.func()
                        self.show()
                if event.type == pygame.QUIT:
                    self.is_running = False
                    break

    def on_click(self, func):
        "Set class parameters equal to parameter and launch event listener thread."
        self.func = func
        listener_thread = threading.Thread(target=self.start_click_listener)
        listener_thread.start()
        listener_thread.join()

    def get_width(self):
        "Returns the width of the button."
        return list(self.size.values())[list(self.size.keys()).index("width")]

    def get_height(self):
        "Returns the height of the button."
        return list(self.size.values())[list(self.size.keys()).index("height")]

    def set_running_status(self, new_state):
        "Updates the is_running state."
        self.is_running = new_state

    def set_position(self, new_position):
        "Updates position of button"
        self.position = new_position
