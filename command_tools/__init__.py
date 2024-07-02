# -*- coding: utf-8 -*-
__author__ = "C418____11 <553515788@qq.com>"
__version__ = "R 1.0.0.4"

# 版本名称规则
# R 正式版
# B | A 测试版3
# T 临时版

"""

一个简单的命令行工具包

"""

from sys import winver as __ver

from .command import *
from .errors import *
from .types import *

__RUN_VERSION = 3.12
if float(__ver) < __RUN_VERSION:
    raise ImportError("Python version Error (at lease {0} now {1})".format(__RUN_VERSION, __ver))
