
from os import environ
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'  # noqa
import pygame

import os

from PIL import Image as im  # Import library that reads image array

from cuticle_analysis import const


class ImageViewer():
    def __init__(self, surface, image_list, dataset_type, position, size):
        self.surface = surface
        self.id = 1
        self.image_list = image_list
        self.image_arr = im.fromarray(image_list.get_image(self.id))
        self.dataset_type = dataset_type
        self.cache_path = f'./dataset/iv_cache.jpg'
        self.image_arr.save(self.cache_path)
        self.image = pygame.image.load(self.cache_path)
        self.img_size = None
        self.position = position
        self.bkg_color = (50, 50, 50)
        self.bkg_obj = pygame.Rect(position[0], position[1], size[0], size[1])
        self.size = size
        self.increment_id = False
        self.decrement_id = False

    def __scale_image__(self):
        """
            Scales the ant image within the image viewer canvas.
            Pre-condition: The class' image variable must be defined.
            Post-condition: An img-ratio is defined and the image is scaled
            adhering to that ratio.
        """
        img_ratio = None

        if self.image.get_width() > self.image.get_height():
            img_ratio = int(self.image.get_width()/self.image.get_height())
            img_height = self.size[0] * img_ratio
            self.image = pygame.transform.scale(
                self.image, (self.size[0], img_height))
        else:
            img_ratio = int(self.image.get_height()/self.image.get_width())
            img_width = self.size[1] * img_ratio
            self.image = pygame.transform.scale(
                self.image, (img_width, self.size[1]))

    def __show__(self):
        "Sets the image to visible"
        pygame.draw.rect(self.surface, self.bkg_color, self.bkg_obj)

        if self.image_arr is not None:
            self.image = pygame.image.load(self.cache_path)
            self.__scale_image__()
            self.surface.blit(self.image, self.position)
        else:
            body_font = pygame.font.SysFont('Arial', 16)
            error_text = body_font.render(
                "Could not locate image '" + f'./{const.DS_MAP[self.dataset_type]}_dataset/{self.id}.jpg' + "'.", True, (255, 255, 255))
            self.surface.blit(
                error_text, (self.position[0], self.position[1] + self.size[1]/2))

    def __get_increment_flag__(self):
        "Returns the increment id flag for the main thread to increment image id."
        return self.increment_id

    def __get_decrement_flag__(self):
        "Returns the increment id flag for the main thread to decrement image id."
        return self.decrement_id

    def __set_increment_flag__(self, new_flag):
        "Sets the increment id flag for the main thread to increment image id."
        self.increment_id = new_flag

    def __set_decrement_flag__(self, new_flag):
        "Sets the decrement id flag for the main thread to decrement image id."
        self.decrement_id = new_flag

    def __update_image__(self, id):
        """
            Updates the image viewer to current image, which is an image cache.
            Post-condition: Ant image is shown in image viewer. If file not found,
            The cache image is deleted and the image viewer shows error.
        """
        if id >= 1 and id <= 1773:
            self.id = id
            try:
                self.image_arr = im.fromarray(
                    self.image_list.get_image(self.id))
                self.image_arr.save(self.cache_path)
                return 1
            except Exception as e:
                print(e)
                self.image_arr = None
                self.__delete_img_cache__()
                return -1
        return 0

    def __delete_img_cache__(self):
        "Deletes the image cache file upon program exit."
        os.remove(self.cache_path)
        self.image = None
