# -*- coding: utf-8 -*-
__author__ = "C418____11 <553515788@qq.com>"

from functools import update_wrapper
from functools import wraps
from typing import Callable
from typing import Never
from typing import Optional
from typing import Union

from . import errors
from . import types

DefaultCommandList = types.CommandList()
_default_lead_char = types.LeadChar(
    ['!', '！',
     '#',
     '$', '￥',
     '//', '/',
     '\\\\', '\\',
     ':', '：'
     ]
)  # 利用python浅拷贝特性完成节省内存,所以套了N层


def default_cut_rule(string: any, *_, **__):
    if type(string) is not str:
        return string
    return string.split()


class Command:
    def __init__(
            self,
            name: Union[str, float],
            op_level: Union[types.OperateLevel, float] = 0,
            *,
            args_maker: Callable = default_cut_rule,
            cut_rule: Optional[Callable[[str], list[str]]] = None,
            lead_char: Optional[types.LeadChar] = None,
            description: str = "Not defined!",
            usage: str = "Not defined!",
            cmd_list: Optional[types.CommandList] = None
    ):

        """

        用于注册指令

        :param name: 指令名
        :type name: str
        :param op_level: 需求权限等级
        :type op_level: Union[types.OperateLevel, float]
        :param args_maker: 对参数修改的函数
        :type args_maker: Callable
        :param cut_rule: 裁剪规则
        :type cut_rule: Optional[Callable[[str], list[str]]]
        :param lead_char: 领导符
        :type lead_char: Optional[types.LeadChar]
        :param description: 指令的描述
        :type description: str
        :param usage: 指令的用法 (如果第一个字符是%, 则自动添加指令名)
        :type usage: str
        :param cmd_list: 指令注册表
        :type cmd_list: Optional[types.CommandList]
        """

        if cmd_list is None:
            cmd_list = DefaultCommandList
        if usage.startswith("%"):
            usage = usage.replace("%", f"{name}", 1)

        self._name = name
        self._op_level = op_level
        self._args_maker = args_maker
        self._cut_rule = cut_rule
        self._lead_char = lead_char
        self._help_str = description
        self._cmd_list = cmd_list
        self._append_to_list = False

        self._data = {"description": self._help_str,
                      "usage": usage,
                      "lead_char": self._lead_char,
                      "op_level": self._op_level,
                      "cut_rule": self._cut_rule,
                      "args_maker": self._args_maker}

    def __call__(self, func):

        if not self._append_to_list:  # 如果该func未尝试加入过指令列表
            self._append_to_list = True  # 设置标记位为已尝试加入过列表

            update_wrapper(func, self)  # 更新类装饰器到函数

            try:
                self._cmd_list[self._name]  # 如果有同名指令已在列表(程序将跳转至else代码块
            except KeyError:  # 如果该指令尚未注册(指令不存在
                self._cmd_list[self._name] = {"func": func, **self._data}  # 注册指令
            else:  # 如果try的代码块能正常运行就会跳转到else(指令存在
                raise errors.CommandAlreadyExistError(command_name=self._name)  # 指令已存在,进行报错

        @wraps(func)
        def decorated(*args, **kwargs):
            return func(*args, **kwargs)

        return decorated


def default_args_unpacker(*args, func, **kwargs):
    return func(*args, **kwargs)


class RunCommand:
    def __init__(
            self,
            *,
            lead_char: Optional[types.LeadChar] = _default_lead_char,
            cut_rule: Callable[[str], list[str]] = default_cut_rule,
            args_maker: Callable = default_cut_rule,
            args_unpacker: Optional[Callable] = None,
            cmd_list: Optional[types.CommandList] = None,
    ):

        """

        用于运行指令

        :param lead_char: 指令前缀
        :type lead_char: Optional[types.LeadChar]
        :param cut_rule: 指令切割规则
        :type cut_rule: Callable[[str], list[str]]
        :param args_maker: 指令参数生成器
        :type args_maker: Callable
        :param args_unpacker: 指令参数解包器
        :type args_unpacker: Optional[Callable]
        :param cmd_list: 指令列表
        :type cmd_list: Optional[types.CommandList]
        """
        if args_unpacker is None:
            args_unpacker = default_args_unpacker
        if cmd_list is None:
            cmd_list = DefaultCommandList

        self._lead_char = lead_char
        self._cut_rule = cut_rule
        self._args_maker = args_maker
        self._args_unpacker = args_unpacker
        self._cmd_list = cmd_list

    def _clear_lead_char(self, string: str, lead_char: Optional[types.LeadChar] = Never) -> str:
        """
        清除领导符

        :param string: 原指令
        :type string: str
        :param lead_char: 领导符列表
        :type lead_char: Union[types.LeadChar, None]
        :return: 根指令
        :rtype: str
        """
        if lead_char is Never:
            lead_char = self._lead_char
        if lead_char is None:
            return string

        for char in lead_char:
            if string[:len(char)] != char:
                continue

            return string[len(char):]
        raise errors.LeadCharNotFindError(
            f"lead char {lead_char} not find in {string}"
        )

    def _get_command_obj(
            self,
            string: str,
            cut_rule: Optional[Callable[[str], list[str]]] = None
    ) -> Union[dict, None]:
        """

        获取指令对象

        :param string: 指令名
        :type string: str
        :param cut_rule: 指令切割规则
        :type cut_rule: Callable[[str], list[str]]
        :return: 指令对应的对象
        :rtype: Union[bool, dict]
        """
        if cut_rule is None:
            cut_rule = self._cut_rule

        try:
            first_word = cut_rule(string)[0]  # 获取根指令
        except IndexError:
            return None

        cmds = self._cmd_list.keys()
        if first_word in cmds:  # 查找指令
            return self._cmd_list[first_word]  # 返回找到dict的对象
        return None

    def run_by_str(self, string: str, op_level: Union[types.OperateLevel, float], *args, **kwargs):
        """
        以字符串运行指令

        :param string: 原始指令字符串
        :type string: str
        :param op_level: 权限等级
        :type op_level: Union[OperateLevel, float]
        :param args: 额外向指令的位置参数
        :param kwargs: 额外向指令的关键字参数
        :return: 指令运行后返回的返回值
        :rtype: Any
        """
        no_lead_char = self._clear_lead_char(string)
        cmd_obj = self._get_command_obj(no_lead_char)
        if cmd_obj is None:
            raise errors.CommandNotFindError(command_name=string)

        self._clear_lead_char(string, cmd_obj["lead_char"])
        temp_obj = self._get_command_obj(no_lead_char, cmd_obj["cut_rule"])
        if cmd_obj != temp_obj:
            raise errors.CommandNotFindError(command_name=string)

        if op_level < cmd_obj["op_level"]:
            raise errors.DontHavePermissionError(level_need=cmd_obj["op_level"], level_now=op_level)

        global_maker_result = self._args_maker(string=string, cmd_obj=cmd_obj, *args, **kwargs)

        local_maker = cmd_obj["args_maker"]
        local_maker_result = local_maker(string=global_maker_result, cmd_obj=cmd_obj, *args, **kwargs)

        ret = self._args_unpacker(local_maker_result, func=cmd_obj["func"])

        return ret

    def __call__(self, string: str, op_level: Union[types.OperateLevel, float], *args, **kwargs):

        """

        运行run_by_str的快捷方式

        以字符串运行指令

        :param string: str: 原始指令字符串
        :param op_level: Union[OperateLevel, float]: 权限等级
        :param args: 额外向指令的位置参数
        :param kwargs: 额外向指令的关键字参数
        :return object: 指令运行后返回的返回值

        """

        return self.run_by_str(string=string, op_level=op_level, *args, **kwargs)


__all__ = (
    "Command",
    "RunCommand",

    "DefaultCommandList",
)
