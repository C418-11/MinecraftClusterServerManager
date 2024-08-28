# -*- coding: utf-8 -*-
# cython: language_level = 3

__author__ = "C418____11 <553515788@qq.com>"
__version__ = "0.0.1"

import functools
import inspect
import os.path
from abc import ABC
from abc import abstractmethod
from copy import deepcopy
from enum import Enum
from types import ModuleType
from types import UnionType
from typing import Any
from typing import Self
from typing import Sequence
from typing import TypeVar


class ConfigOperate(Enum):
    Delete = "Delete"
    Read = "Read"
    Write = "Write"
    Unknown = None


class RequiredKeyNotFoundError(KeyError):
    def __init__(self, key: str, current_key: str, index: int, operate: ConfigOperate = ConfigOperate.Unknown):
        super().__init__(current_key)

        self.key = key
        self.current_key = current_key
        self.index = index
        self.operate = operate

    def __str__(self):
        string = f"{self.key} -> {self.current_key} ({self.index + 1} / {len(self.key.split('.'))})"
        if self.operate.value is not None:
            string += f" Operate: {self.operate.value}"
        return string


class ConfigDataTypeError(TypeError):
    def __init__(self, key: str, current_key: str, index: int, required_type: type[object], now_type: type[object]):
        super().__init__(current_key)

        self.key = key
        self.current_key = current_key
        self.index = index
        self.requited_type = required_type
        self.now_type = now_type

    def __str__(self):
        return (
            f"{self.key} -> {self.current_key} ({self.index + 1} / {len(self.key.split('.'))})"
            f" Must be '{self.requited_type}'"
            f", Not '{self.now_type}'"
        )


class UnsupportedConfigFormatError(Exception):
    def __init__(self, _format: str):
        super().__init__(f"Unsupported config format: {_format}")
        self.format = _format


def _norm_join(*paths: str) -> str:
    return os.path.normpath(os.path.join(*paths))


def _is_method(func):
    arguments = inspect.getargs(func.__code__).args
    if len(arguments) < 1:
        return False
    return arguments[0] in {"self", "cls"}


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
            if not isinstance(now_data, dict):
                raise ConfigDataTypeError(path, now_path, path_index, dict, type(now_data))
            if now_path not in now_data:
                raise RequiredKeyNotFoundError(path, now_path, path_index, ConfigOperate.Read)

        def process_return(now_data):
            if get_raw and type(now_data) is dict:
                return ConfigData(deepcopy(now_data))

            return deepcopy(now_data)

        return self._process_path(path, checker, process_return)

    def setPathValue(self, path: str, value: Any, *, allow_create: bool = True) -> None:
        def checker(now_data, now_path, last_path, path_index):
            if not isinstance(now_data, dict):
                raise ConfigDataTypeError(path, now_path, path_index, dict, type(now_data))
            if now_path not in now_data:
                if not allow_create:
                    raise RequiredKeyNotFoundError(path, now_path, path_index, ConfigOperate.Write)
                now_data[now_path] = {}

            if last_path is None:
                now_data[now_path] = value

        self._process_path(path, checker, lambda *_: None)

    def deletePath(self, path: str) -> None:
        def checker(now_data, now_path, last_path, path_index):
            if not isinstance(now_data, dict):
                raise ConfigDataTypeError(path, now_path, path_index, dict, type(now_data))
            if now_path not in now_data:
                raise RequiredKeyNotFoundError(path, now_path, path_index, ConfigOperate.Delete)

            if last_path is None:
                del now_data[now_path]
                return True

        self._process_path(path, checker, lambda *_: None)

    def hasPath(self, path: str) -> bool:
        def checker(now_data, now_path, _last_path, path_index):
            if not isinstance(now_data, dict):
                raise ConfigDataTypeError(path, now_path, path_index, dict, type(now_data))
            if now_path not in now_data:
                return False

        return self._process_path(path, checker, lambda *_: True)

    def get(self, path, default=None):
        try:
            return self.getPathValue(path)
        except RequiredKeyNotFoundError:
            return default

    def keys(self):
        return self._data.keys()

    def values(self):
        return [(ConfigData(x) if type(x) is dict else x) for x in self._data.values()]

    def items(self):
        return [(k, (ConfigData(v) if type(v) is dict else v)) for k, v in self._data.items()]

    def __getitem__(self, key):
        return self.getPathValue(key)

    def __setitem__(self, key, value):
        self.setPathValue(key, value)

    def __delitem__(self, key):
        self.deletePath(key)

    def __contains__(self, key):
        return self.hasPath(key)

    def __getattr__(self, item):
        return ConfigData(self._data[item]) if type(self._data[item]) is dict else self._data[item]

    def __iter__(self):
        return iter(self._data)

    def __repr__(self):
        data_str = f"{self._data!r}"[1:-1]

        return f"{self.__class__.__name__}({data_str})"

    def __deepcopy__(self, memo):
        return ConfigData(deepcopy(self._data, memo))


class RequiredKey:
    AllowType = {str, int, float, bool, list, dict, tuple}
    TypingType = {UnionType}

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
            if (type(default) not in self.TypingType) and (type(default) is not type):
                _type = type(default)
                value = deepcopy(default)
                try:
                    value = data.getPathValue(path)
                except RequiredKeyNotFoundError:
                    if allow_create:
                        data.setPathValue(path, value, allow_create=True)
            else:
                value = data.getPathValue(path)

            if (_type is dict) and isinstance(value, ConfigData):
                value = value.data

            if not isinstance(value, _type):
                path_chunks = path.split('.')
                raise ConfigDataTypeError(path, path_chunks[-1], len(path_chunks) - 1, _type, type(value))

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
        self.SLProcessor: dict[str, ABCConfigSL] = {}
        self.FileExtProcessor: dict[str, set[str]] = {}


class ABCConfig(ABC):

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

    @abstractmethod
    def save(
            self,
            config_pool: ABCConfigPool,
            namespace: str | None = None,
            file_name: str | None = None,
            config_format: str | None = None,
            *processor_args,
            **processor_kwargs
    ) -> None:
        ...

    @classmethod
    @abstractmethod
    def load(
            cls,
            config_pool: ABCConfigPool,
            namespace: str,
            file_name: str,
            config_format: str,
            *processor_args,
            **processor_kwargs
    ) -> Self:
        ...

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


SLArgument = Sequence | dict | tuple[Sequence, dict[str, Any]]
C = TypeVar("C", bound=ABCConfig)


class ABCConfigSL(ABC):

    def __init__(self, s_arg: SLArgument = None, l_arg: SLArgument = None, create_dir: bool = True):
        def _build_arg(value) -> tuple[list, dict[str, Any]]:
            if value is None:
                return [], {}
            if isinstance(value, Sequence):
                return list(value), {}
            if isinstance(value, dict):
                return [], value
            raise TypeError(f"Invalid argument type, must be '{SLArgument}'")

        self.save_arg: tuple[list, dict[str, Any]] = _build_arg(s_arg)
        self.load_arg: tuple[list, dict[str, Any]] = _build_arg(l_arg)

        self.create_dir = create_dir

    @property
    @abstractmethod
    def regName(self) -> str:
        ...

    @property
    @abstractmethod
    def fileExt(self) -> list[str]:
        ...

    def registerTo(self, config_pool: ABCConfigPool = None):
        if config_pool is None:
            config_pool = DefaultConfigPool

        config_pool.SLProcessor[self.regName] = self
        for ext in self.fileExt:
            if ext not in config_pool.FileExtProcessor:
                config_pool.FileExtProcessor[ext] = {self.regName}
                continue
            config_pool.FileExtProcessor[ext].add(self.regName)

    @classmethod
    def enable(cls):
        ...

    @abstractmethod
    def save(self, config: ABCConfig, root_path: str, namespace: str, file_name: str, *args, **kwargs) -> None:
        ...

    @abstractmethod
    def load(
            self,
            config_cls: type[C],
            root_path: str,
            namespace: str,
            file_name: str,
            *args,
            **kwargs
    ) -> C:
        ...

    def _getFilePath(
            self,
            config: ABCConfig,
            root_path: str,
            namespace: str = None,
            file_name: str = None,
    ):
        if namespace is None:
            namespace = config.namespace
        if file_name is None:
            file_name = config.file_name

        if namespace is None or file_name is None:
            raise ValueError("namespace and file_name can't be None")

        full_path = _norm_join(root_path, namespace, file_name)
        if self.create_dir:
            os.makedirs(os.path.dirname(full_path), exist_ok=True)

        return full_path


class Config(ABCConfig):

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

        if config_format not in config_pool.SLProcessor:
            raise UnsupportedConfigFormatError(config_format)

        return config_pool.SLProcessor[config_format].save(
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

        if config_format not in config_pool.SLProcessor:
            raise UnsupportedConfigFormatError(config_format)

        return config_pool.SLProcessor[
            config_format
        ].load(
            cls,
            config_pool.root_path,
            namespace,
            file_name,
            *processor_args,
            **processor_kwargs
        )


yaml: ModuleType


class YamlSL(ABCConfigSL):
    @property
    def regName(self) -> str:
        return "yaml"

    @property
    def fileExt(self) -> list[str]:
        return [".yaml"]

    @classmethod
    def enable(cls):
        global yaml
        import yaml
        yaml = yaml

    def save(self, config: ABCConfig, root_path: str, namespace: str, file_name: str, *args, **kwargs) -> None:
        new_args = deepcopy(self.save_arg[0])[:len(args)] = args
        new_kwargs = deepcopy(self.save_arg[1]) | kwargs

        file_path = self._getFilePath(config, root_path, namespace, file_name)
        with open(file_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config.data.data, f, *new_args, **new_kwargs)

    def load(
            self,
            config_cls: type[C],
            root_path: str,
            namespace: str,
            file_name: str,
            *args,
            **kwargs
    ) -> C:
        with open(_norm_join(root_path, namespace, file_name), "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        obj = config_cls(ConfigData(data))
        obj.namespace = namespace
        obj.file_name = file_name
        obj.config_format = self.regName

        return obj


json: ModuleType


class JsonSL(ABCConfigSL):

    @property
    def regName(self) -> str:
        return "json"

    @property
    def fileExt(self) -> list[str]:
        return [".json"]

    @classmethod
    def enable(cls):
        global json
        import json
        json = json

    def save(self, config: ABCConfig, root_path: str, namespace: str, file_name: str, *args, **kwargs) -> None:
        new_args = deepcopy(self.save_arg[0])[:len(args)] = args
        new_kwargs = deepcopy(self.save_arg[1]) | kwargs

        file_path = self._getFilePath(config, root_path, namespace, file_name)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(config.data.data, f, *new_args, **new_kwargs)

    def load(
            self,
            config_cls: type[C],
            root_path: str,
            namespace: str,
            file_name: str,
            *args,
            **kwargs
    ) -> C:
        new_args = deepcopy(self.load_arg[0])[len(args)] = args
        new_kwargs = deepcopy(self.load_arg[1]) | kwargs

        with open(_norm_join(root_path, namespace, file_name), "r", encoding="utf-8") as f:
            data = json.load(f, *new_args, **new_kwargs)

        obj = config_cls(ConfigData(data))
        obj.namespace = namespace
        obj.file_name = file_name
        obj.config_format = self.regName

        return obj


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
            raw_file_name: str,
            required: RequiredKey,
            *,
            config_format: str = None,
            allow_create: bool = True,
    ):
        if config_format is None:
            _, config_format = os.path.splitext(raw_file_name)
            if not config_format:
                raise UnsupportedConfigFormatError("Unknown")
            if config_format not in config_pool.FileExtProcessor:
                raise UnsupportedConfigFormatError(config_format)
            config_format = next(iter(config_pool.FileExtProcessor[config_format]))

        if config_format not in config_pool.SLProcessor:
            raise UnsupportedConfigFormatError(config_format)

        config: ABCConfig | None = config_pool.get(namespace, raw_file_name)
        if config is None:
            try:
                config = Config.load(config_pool, namespace, raw_file_name, config_format)
            except FileNotFoundError:
                if not allow_create:
                    raise
                config = Config(ConfigData({}))
                config.namespace = namespace
                config.file_name = raw_file_name
                config.config_format = config_format

            config_pool.set(namespace, raw_file_name, config)

        self._config: ABCConfig = config
        self._required = required

        self._allow_create = allow_create

    def checkConfig(self):
        return self._required.filter(self._config.data, allow_create=self._allow_create)

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

    def _method_processor(self, obj, *args):
        return obj, self._required.filter(self._config.data, allow_create=self._allow_create), *args


DefaultConfigPool = ConfigPool()
requireConfig = DefaultConfigPool.requireConfig

__all__ = (
    "RequiredKeyNotFoundError",
    "ConfigDataTypeError",
    "UnsupportedConfigFormatError",

    "ConfigData",
    "RequiredKey",
    "ABCConfig",
    "ABCConfigSL",
    "Config",
    "YamlSL",
    "JsonSL",
    "ConfigPool",
    "RequireConfigDecorator",

    "DefaultConfigPool",
    "requireConfig",
)
