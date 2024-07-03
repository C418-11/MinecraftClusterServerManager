# -*- coding: utf-8 -*-
# cython: language_level = 3

__author__ = "C418____11 <553515788@qq.com>"
__version__ = "0.1"

import sys

import colorama


def file_is_empty(func):
    def wrapper(self, *args, **kwargs):
        if self.super_ is None:
            return
        return func(self, *args, **kwargs)

    return wrapper


class ColorWrite:
    """
    Output to the specified color
    """

    def __init__(self, super_, font_color=None, bg_color=None):
        colorama.init()
        self.super_ = super_
        self.font_color = font_color
        self.bg_color = bg_color

    @file_is_empty
    def write(self, text):
        pre = ""
        if self.font_color is not None:
            pre += self.font_color
        if self.bg_color is not None:
            pre += self.bg_color
        self.super_.write(pre + text + colorama.Style.RESET_ALL)

    @file_is_empty
    def flush(self):
        self.super_.flush()

    def __getattribute__(self, item):
        try:
            return object.__getattribute__(self, item)
        except AttributeError:
            try:
                return object.__getattribute__(self.super_, item)
            except AttributeError:
                pass
            raise


sys.stderr = ColorWrite(sys.__stderr__, colorama.Fore.RED)

__all__ = ("ColorWrite",)
