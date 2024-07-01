# -*- coding: utf-8 -*-
# cython: language_level = 3

__author__ = "C418____11 <553515788@qq.com>"
__version__ = "0.0.1Dev"

import os.path
from typing import Any
from typing import Self
import functools
from copy import deepcopy
from typing import Type
import yaml
from typing import Callable
from abc import ABC
import inspect


class RequiredKeyNotFoundError(KeyError):
    def __init__(self, key: str, current_key: str, index: int):
        super().__init__(current_key)

        self.key = key
        self.current_key = current_key
        self.index = index

    def __str__(self):
        return f"{self.key} -> {self.current_key} ({self.index + 1} / {len(self.key.split('.'))})"


class UnsupportedConfigFormatError(Exception):
    def __init__(self, _format: str):
        super().__init__(f"Unsupported config format: {_format}")
        self.format = _format


def norm_join(*paths: str) -> str:
    return os.path.normpath(os.path.join(*paths))


def _is_method(func):
    arguments = inspect.getargs(func.__code__).args
    if len(arguments) < 1:
        return False
    return arguments[0] == "self"


class ConfigData:
    def __init__(self, data: dict):
        self._data = deepcopy(data)

    @property
    def data(self) -> dict:
        return deepcopy(self._data)

    def _process_path(self, path: str, process_check, process_return) -> Any:
        last_path = path
        now_data = self._data

        path_index = -1

        while last_path:
            path_index += 1
            try:
                now_path, last_path = last_path.split('.', maxsplit=1)
            except ValueError:
                now_path, last_path = last_path, None

            check_result = process_check(now_data, now_path, last_path, path_index)
            if check_result is not None:
                return check_result

            now_data = now_data[now_path]

        return process_return(now_data)

    def getPathValue(self, path: str, *, get_raw: bool = True) -> Any:
        def checker(now_data, now_path, _last_path, path_index):
            if now_path not in now_data:
                raise RequiredKeyNotFoundError(path, now_path, path_index)

        def process_return(now_data):
            if get_raw and type(now_data) is dict:
                return ConfigData(deepcopy(now_data))

            return deepcopy(now_data)

        return self._process_path(path, checker, process_return)

    def setPathValue(self, path: str, value: Any, *, allow_create: bool = True) -> None:
        def checker(now_data, now_path, last_path, path_index):
            if now_path not in now_data:
                if not allow_create:
                    raise RequiredKeyNotFoundError(path, now_path, path_index)
                now_data[now_path] = {}

            if last_path is None:
                now_data[now_path] = value

        self._process_path(path, checker, lambda *_: None)

    def deletePath(self, path: str) -> None:
        def checker(now_data, now_path, last_path, path_index):
            if now_path not in now_data:
                raise RequiredKeyNotFoundError(path, now_path, path_index)

            if last_path is None:
                del now_data[now_path]

        self._process_path(path, checker, lambda *_: None)

    def hasPath(self, path: str) -> bool:
        def checker(now_data, now_path, *_):
            if now_path not in now_data:
                return False

        return self._process_path(path, checker, lambda *_: True)

    def __getitem__(self, key):
        return self.getPathValue(key)

    def __setitem__(self, key, value):
        self.setPathValue(key, value)

    def __delitem__(self, key):
        self.deletePath(key)

    def __contains__(self, key):
        return self.hasPath(key)

    def __repr__(self):
        data_str = f"{self._data!r}"[1:-1]

        return f"{self.__class__.__name__}({data_str})"

    def __deepcopy__(self, memo):
        return ConfigData(deepcopy(self._data, memo))


class RequiredKey:
    AllowType = {str, int, float, bool, list, dict, tuple}

    def __init__(self, paths: list[str] | dict[str, Any]):

        self._check_type: bool = type(paths) is dict
        self._paths: list | dict[str, type] = paths

        if self._check_type and (not self.check()):
            raise TypeError("Invalid type for paths")

    def filter(self, data: ConfigData, *, allow_create: bool = False):
        result = ConfigData({})

        if not self._check_type:
            for path in self._paths:
                value = data.getPathValue(path)
                result[path] = value
            return result

        for path, default in self._paths.items():

            _type = default
            if type(default) is not type:
                _type = type(default)
                value = default
                try:
                    value = data.getPathValue(path)
                except RequiredKeyNotFoundError:
                    if allow_create:
                        data.setPathValue(path, value, allow_create=True)
            else:
                value = data.getPathValue(path)

            if (_type is dict) and isinstance(value, ConfigData):
                value = value.data

            if type(value) is not _type:
                raise TypeError(f"Path {path} is not {_type.__name__}")

            result[path] = value

        return result

    def check(self) -> bool:
        types = {k: v for k, v in self._paths.items() if type(v) is type}
        if any(v not in self.AllowType for v in types.values()):
            return False
        return True


class ABCConfigPool(ABC):
    def __init__(self, root_path: str = "./.config"):
        self.root_path = root_path


class ABCConfig(ABC):
    SaveProcessor: dict[str, Callable[[Self, str, str, str, Any], None]] = {}
    LoadProcessor: dict[str, Callable[[Type[Self], str, str, str, Any], ConfigData]] = {}

    def __init__(
            self,
            config_data: ConfigData,
            *,
            namespace: str = None,
            file_name: str = None,
            config_format: str = None
    ) -> None:
        self.data: ConfigData = config_data

        self.namespace: str | None = namespace
        self.file_name: str | None = file_name
        self.config_format: str | None = config_format

    def save(
            self,
            config_pool: ABCConfigPool,
            namespace: str | None = None,
            file_name: str | None = None,
            config_format: str | None = None,
            *processor_args,
            **processor_kwargs
    ) -> None:
        if config_format is None:
            config_format = self.config_format

        if config_format is None:
            raise ValueError("file_name and config_format can't be None")

        if config_format not in self.SaveProcessor:
            raise UnsupportedConfigFormatError(config_format)

        return self.SaveProcessor[config_format](
            self,
            config_pool.root_path,
            namespace,
            file_name,
            *processor_args,
            **processor_kwargs
        )

    @classmethod
    def load(
            cls,
            config_pool: ABCConfigPool,
            namespace: str,
            file_name: str,
            config_format: str,
            *processor_args,
            **processor_kwargs
    ) -> Self:
        if config_format not in cls.LoadProcessor:
            raise UnsupportedConfigFormatError(config_format)

        func = getattr(cls, cls.LoadProcessor[config_format].__name__)
        return func(config_pool.root_path, namespace, file_name, *processor_args, **processor_kwargs)

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.data[key] = value

    def __delitem__(self, key: str) -> None:
        del self.data[key]

    def __contains__(self, key: str) -> bool:
        return key in self.data

    def __repr__(self):
        field_str: str = ''
        for field in ["config_format", "data", "namespace", "file_name"]:
            field_value = getattr(self, field)
            if field_value is None:
                continue

            field_str += f", {field}={field_value!r}"

        field_str = field_str[2:]

        return f"{self.__class__.__name__}({field_str})"


class Config(ABCConfig):
    @classmethod
    def from_yaml(cls, root_path: str, namespace: str, file_name: str) -> Self:
        with open(norm_join(root_path, namespace, file_name), "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        obj = cls(ConfigData(data))
        obj.namespace = namespace
        obj.file_name = file_name
        obj.config_format = "yaml"

        return obj

    def save_yaml(self, root_path: str, namespace: str, file_name: str, *args, **kwargs) -> None:
        file_path = self._file_path(root_path, namespace, file_name)
        with open(file_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.data.data, f, *args, **kwargs)

    SaveProcessor = {
        "yaml": save_yaml
    }
    LoadProcessor = {
        "yaml": from_yaml
    }

    def _file_path(self, root_path, namespace: str = None, file_name: str = None, /, *, create_dir: bool = True):
        if namespace is None:
            namespace = self.namespace
        if file_name is None:
            file_name = self.file_name

        if namespace is None or file_name is None:
            raise ValueError("namespace and file_name can't be None")

        full_path = norm_join(root_path, namespace, file_name)
        if create_dir:
            os.makedirs(os.path.dirname(full_path), exist_ok=True)

        return full_path


class ConfigPool(ABCConfigPool):
    def __init__(self, root_path="./.config"):
        super().__init__(root_path)
        self._configs: dict[str, dict[str, ABCConfig]] = {}

    def get(self, namespace: str, file_name: str = None, /):
        result = {}
        if namespace in self._configs:
            result = self._configs[namespace]

        if file_name is None:
            return result

        if file_name in result:
            return result[file_name]

        return None

    def set(self, namespace: str, file_name: str, config: ABCConfig):
        if namespace not in self._configs:
            self._configs[namespace] = {}

        self._configs[namespace][file_name] = config

    def saveAll(self):
        for configs in self._configs.values():
            for config in configs.values():
                config.save(self)

    def requireConfig(
            self,
            namespace: str,
            file_name: str,
            required: list[str] | dict[str, Any],
            **kwargs,
    ):
        return RequireConfigDecorator(self, namespace, file_name, RequiredKey(required), **kwargs)

    @property
    def configs(self):
        return deepcopy(self._configs)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.configs!r})"


class RequireConfigDecorator:
    def __init__(
            self,
            config_pool: ConfigPool,
            namespace: str,
            file_name: str,
            required: RequiredKey,
            *,
            config_format: str = None,
            allow_create: bool = True,
            config_class: ABCConfig = Config
    ):
        if config_format is None:
            if '.' not in file_name:
                raise UnsupportedConfigFormatError("Unknown")
            config_format = file_name.split('.')[-1]

        if config_format not in config_class.LoadProcessor:
            raise UnsupportedConfigFormatError(config_format)

        config: ABCConfig | None = config_pool.get(namespace, file_name)
        if config is None:
            try:
                config = Config.load(config_pool, namespace, file_name, config_format)
            except FileNotFoundError:
                if not allow_create:
                    raise
                config = Config(ConfigData({}))
                config.namespace = namespace
                config.file_name = file_name
                config.config_format = config_format

            config_pool.set(namespace, file_name, config)

        self._config: ABCConfig = config
        self._required = required

        self._allow_create = allow_create

    def __call__(self, func):
        if _is_method(func):
            processor = self._method_processor
        else:
            processor = self._function_processor

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*processor(*args), **kwargs)

        return wrapper

    def _function_processor(self, *args):
        return self._required.filter(self._config.data, allow_create=self._allow_create), *args

    def _method_processor(self, self_obj, *args):
        return self_obj, self._required.filter(self._config.data, allow_create=self._allow_create), *args


DefaultConfigPool = ConfigPool()
requireConfig = DefaultConfigPool.requireConfig


__all__ = (
    "RequiredKeyNotFoundError",
    "UnsupportedConfigFormatError",

    "ConfigData",
    "RequiredKey",
    "ABCConfig",
    "Config",
    "ConfigPool",
    "RequireConfigDecorator",

    "DefaultConfigPool",
    "requireConfig",
)
