# -*- coding: utf-8 -*-
__author__ = "C418____11 <553515788@qq.com>"

"""
You must import command_tools.errors first !!!

else

you might get a error like this

AttributeError: partially initialized module 'command_tools.types_' has no attribute 'OperateLevel' (most likely due 
to a circular import)
"""


import pickle
from typing import Union, Tuple

import numbers

from command_tools import errors


class CommandList:
    """
    指令列表
    """

    def __init__(self, list_: dict = None):
        """
        :param list_: 已存在的指令列表
        """
        if list_ is None:
            list_ = {}

        self.list_ = list_

    def keys(self):
        return self.list_.keys()

    def values(self):
        return self.list_.values()

    def __getitem__(self, item):
        return self.list_[item]

    def __setitem__(self, key, value):
        self.list_[key] = value


class LeadChar:
    """
    领导符列表
    """

    def __init__(self, list_: list):
        self.list_ = list_

    def __getitem__(self, item):
        return self.list_[item]


class OperateLevel:
    """
    权限等级
    """

    def __init__(self, name: object, level: numbers.Real) -> None:
        self.name = name
        self.level = level

    def __getitem__(self, item):
        return object.__getattribute__(self, item)

    def __call__(self):
        return self.name, self.level

    def __str__(self):
        return f"{self.name}: {self.level}"

    def __repr__(self):
        return f"OperateLevel(name={self.name}, level={self.level})"

    def __int__(self):
        return int(self.level)

    def __float__(self):
        return float(self.level)

    def __abs__(self):
        return abs(self.level)

    def __bool__(self):
        return bool(self.level)

    def __eq__(self, other):
        return self.level == other.level

    def __ge__(self, other):
        return self.level >= other

    def __gt__(self, other):
        return self.level > other

    def __le__(self, other):
        return self.level <= other

    def __lt__(self, other):
        return self.level < other


class OperateLevelList:
    """
    权限等级注册表
    """

    def __init__(self):
        self.level_list = {}

    def append(self, op_level: Union[OperateLevel, Tuple[object, numbers.Real]]) -> None:
        """
        :param op_level: 权限等级
        """

        if type(op_level) is tuple:
            level = OperateLevel(*op_level)
        else:
            level = op_level

        try:
            self.level_list[level.name]
        except KeyError:
            self.level_list[level.name] = level
        else:
            raise errors.OperateLevelAlreadyExistError(level_name=level.name)

    def save(self, file) -> None:
        """
        :param file: 文件路径
        """
        if type(file) is str:
            file = open(file, mode="wb")
        pickle.dump(self, file)

    @staticmethod
    def load(file):
        """
        :param file: 文件路径
        """
        if type(file) is str:
            file = open(file, mode="rb")
        return pickle.load(file)

    def __getitem__(self, item) -> OperateLevel:
        return self.level_list[item]


class UserList:
    """
    用户列表
    """

    def __init__(self, default_level: Union[OperateLevel] = OperateLevel("default", 0)):
        """
        :param default_level: 默认权限等级
        """
        self.user_list = {}
        self.default_level = default_level

    def append(self, user_name, op_level: OperateLevel):
        """
        :param user_name: 用户名
        :param op_level: 权限等级
        """
        try:
            self.user_list[user_name]
        except KeyError:
            self.user_list[user_name] = op_level
        else:
            raise errors.UserAlreadyExistError(user_name=user_name)

    def __getitem__(self, item) -> OperateLevel:
        try:
            return self.user_list[item]
        except KeyError:
            return self.default_level

    def __setitem__(self, key, value):
        if isinstance(value, numbers.Real):
            value = OperateLevel(value, value)
        self.user_list[key] = value

    def reset_level(self, user_name, op_level: OperateLevel):
        """
        :param user_name: 用户名
        :param op_level: 权限等级
        """
        self.user_list[user_name] = op_level

    def save(self, file):
        """
        :param file: 文件路径
        """
        if type(file) is str:
            file = open(file, mode="wb")
        pickle.dump(self, file)

    @classmethod
    def load(cls, file):
        """
        :param file: 文件路径
        """

        file = open(file, mode="rb")
        try:
            return pickle.load(file)
        except EOFError:
            raise FileNotFoundError
