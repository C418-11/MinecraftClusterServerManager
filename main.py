# -*- coding: utf-8 -*-
# cython: language_level = 3

__author__ = "C418____11 <553515788@qq.com>"
__version__ = "0.0.1Dev"

import codecs
import functools
import inspect
import os
import subprocess
import sys
import threading
import time
import traceback
from copy import deepcopy
from itertools import zip_longest
from platform import system
from typing import Any
from typing import Callable
from typing import Generator
from weakref import WeakKeyDictionary

import colorama

from StdColor import ColorWrite
from buffer import StringBuffer
from command_tools import Command
from command_tools import CommandException
from command_tools import DefaultCommandList
from command_tools import RunCommand
from config import ConfigData
from config import DefaultConfigPool
from config import SimpleYamlSL
from config import requireConfig

SimpleYamlSL.enable()
SimpleYamlSL().registerTo()

default_config = {
    "process": {
        "DefaultEncoding.stdin": "utf-8",
        "DefaultEncoding.stdout": "utf-8",
        "Register": {},
    }
}

if system() == "Windows":
    default_config["process"] |= {
        "DefaultEncoding.stdin": "utf-8",
        "DefaultEncoding.stdout": "gbk",
        "Register": {
            "cmd": {
                "description": "windows command line",
                "cmd": "cmd",
                "end": "exit",
                "Encoding": {
                    "stdin": "gbk"
                }
            },
            "powershell": {
                "description": "PowerShell",
                "abbreviation": "ps",
                "cmd": "powershell",
                "end": "exit"
            },
            "ping": {
                "description": "ping localhost",
                "cmd": "ping -l 1145 localhost",
            },
        },
    }
elif system() == "Linux":
    default_config["process"] |= {
        "Register": {
            "bash": {
                "abbreviation": "sh",
                "cmd": "bash",
                "end": "exit"
            }
        },
    }

STDOUT_LIGHTRED = ColorWrite(sys.stdout, colorama.Fore.LIGHTRED_EX)
STDOUT_LIGHTGREEN = ColorWrite(sys.stdout, colorama.Fore.LIGHTGREEN_EX)
STDOUT_LIGHTYELLOW = ColorWrite(sys.stdout, colorama.Fore.LIGHTYELLOW_EX)
STDOUT_LIGHTBLUE = ColorWrite(sys.stdout, colorama.Fore.LIGHTBLUE_EX)
STDOUT_LIGHTMAGENTA = ColorWrite(sys.stdout, colorama.Fore.LIGHTMAGENTA_EX)
STDOUT_LIGHTCYAN = ColorWrite(sys.stdout, colorama.Fore.LIGHTCYAN_EX)

STDOUT_RED = ColorWrite(sys.stdout, colorama.Fore.RED)
STDOUT_GREEN = ColorWrite(sys.stdout, colorama.Fore.GREEN)
STDOUT_YELLOW = ColorWrite(sys.stdout, colorama.Fore.YELLOW)
STDOUT_BLUE = ColorWrite(sys.stdout, colorama.Fore.BLUE)
STDOUT_MAGENTA = ColorWrite(sys.stdout, colorama.Fore.MAGENTA)
STDOUT_CYAN = ColorWrite(sys.stdout, colorama.Fore.CYAN)


def _has_encoding(encoding: str) -> bool:
    try:
        codecs.lookup(encoding)
        return True
    except LookupError:
        return False


class SubprocessService:
    def __init__(self, service_name: str):
        self._name: str = service_name

        process_config: ConfigData = requireConfig(
            '', "process.yaml", {
                f"Register.{service_name}": dict,
                f"Register.{service_name}.cmd": list | str,
                f"Register.{service_name}.workdir": '.',
            }
        ).checkConfig()[f"Register.{service_name}"]

        default_encoding: ConfigData = requireConfig(
            '', "process.yaml", {
                "DefaultEncoding.stdin": str,
                "DefaultEncoding.stdout": str,
            }
        ).checkConfig()["DefaultEncoding"]

        _stdin_encoding: str = process_config.get(
            "Encoding.stdin", default_encoding.get("stdin", "utf-8")
        )
        if not _has_encoding(_stdin_encoding):
            raise ValueError(f"Encoding {_stdin_encoding} not found")
        _stdout_encoding: str = process_config.get(
            "Encoding.stdout", default_encoding.get("stdout", "utf-8")
        )
        if not _has_encoding(_stdout_encoding):
            raise ValueError(f"Encoding {_stdout_encoding} not found")
        self._stdin_encoding: str = _stdin_encoding
        self._stdout_encoding: str = _stdout_encoding

        self._abbreviation: str | None = process_config.get("abbreviation")
        self._description: str | None = process_config.get("description")
        self._start_cmd: str | list[str] = process_config["cmd"]
        self._end_cmd: str | None = process_config.get("end")
        self._workdir: str = process_config["workdir"]
        self.process_config: ConfigData = deepcopy(process_config)

        del self.process_config["cmd"]
        del self.process_config["workdir"]
        if self._abbreviation is not None:
            del self.process_config["abbreviation"]
        if self._description is not None:
            del self.process_config["description"]
        if self._end_cmd is not None:
            del self.process_config["end"]
        if not os.path.exists(self._workdir):
            raise FileNotFoundError(f"Workdir {self._workdir} not found")

        self._running: bool = False
        self._process: subprocess.Popen | None = None

        self._stdout_buffer: StringBuffer = StringBuffer()
        self._thread: threading.Thread | None = None

    def _buffer_loop(self):
        cache: bytes = b''

        def _process_cache() -> bool:
            nonlocal cache
            try:
                txt = cache.decode(self._stdout_encoding)
                cache = b''
            except UnicodeDecodeError as e:
                if len(cache) > 5:
                    txt = '?'
                    cache = cache[:e.start] + cache[e.end:]
                else:
                    return True

            self._stdout_buffer.write(txt)

        while self._running and self._process.poll() is None:
            cache += self._process.stdout.read(1)
            if _process_cache():
                continue

        __temp__cache = cache
        if _process_cache():
            print("$$Decode Failed # TODO")  # TODO
        else:
            print("$$Decoded stdout cache$$ # TODO")  # TODO
        if __temp__cache != cache:
            print(f"$$Diff Cache: {__temp__cache!r} {cache!r}")

        if self._process is None:
            self._running = False
            return
        if self._process.poll() is not None:
            self._running = False
            self._process = None

    def connectStdout(self, callback: Callable[[str], None]) -> None:
        self._stdout_buffer.register(callback)

    def disconnectStdout(self, callback: Callable[[str], None]) -> None:
        self._stdout_buffer.unregister(callback)

    def isConnectedStdout(self, callback: Callable[[str], None]) -> bool:
        return self._stdout_buffer.isRegistered(callback)

    def sendStdin(self, txt: str, flush: bool = True) -> None:
        if self._process.poll() is not None:
            raise RuntimeError("Process not running")

        self._process.stdin.write(txt.encode(self._stdin_encoding))
        if flush:
            self._process.stdin.flush()

    def start(self):
        if self._running:
            raise RuntimeError("Process already running")

        self._process = subprocess.Popen(
            self._start_cmd,
            cwd=self._workdir,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        self._running = True
        self._thread = threading.Thread(target=self._buffer_loop, daemon=True)
        self._thread.start()

    def stop(self):
        if not self._running:
            raise RuntimeError("Process not running")

        self._process.terminate()

    def end(self):
        if not self._running:
            raise RuntimeError("Process not running")

        if self._end_cmd is None or self._end_cmd == "^C":
            self._process.terminate()
        else:
            self.sendStdin(f"{self._end_cmd}\n")

    def join(self, timeout: float | None = None):
        if self._process is not None:
            try:
                self._process.wait(timeout)
            except subprocess.TimeoutExpired as e:
                raise TimeoutError(f"Process '{self._name}' did not end within {timeout} seconds") from e

        if self._thread is not None:
            self._thread.join(timeout)
            if self._thread.is_alive():
                raise TimeoutError(f"Thread '{self._thread.name}' did not join within {timeout} seconds")

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def name(self) -> str:
        return self._name

    @property
    def abbreviation(self) -> str | None:
        return self._abbreviation

    @property
    def description(self) -> str | None:
        return self._description

    @property
    def pid(self) -> int:

        return self._process.pid

    @property
    def running(self) -> bool:
        return self._running

    @property
    def end_cmd(self) -> str:
        return self._end_cmd


processes: dict[str, SubprocessService] = {}


@requireConfig('', "process.yaml", {
    "DefaultEncoding.stdin": str,
    "DefaultEncoding.stdout": str,
    "Register": {}
})
def start_processes(config: ConfigData):
    default_encoding = config["DefaultEncoding"]
    if not _has_encoding(default_encoding.stdin):
        raise ValueError(f"Invalid default encoding for stdin: {default_encoding.stdin}")

    if not _has_encoding(default_encoding.stdout):
        raise ValueError(f"Invalid default encoding for stdout: {default_encoding.stdout}")

    for process_name in config["Register"]:

        if process_name in processes:
            raise ValueError(f"Process '{process_name}' already exists")

        try:
            processes[process_name] = SubprocessService(process_name)
        except Exception as e:
            traceback.print_exception(e)
            print(f"Failed to init process '{process_name}': {e}", file=STDOUT_RED)
            continue

        p_abbreviation = processes[process_name].abbreviation
        if p_abbreviation is None:
            print(f"Registered process '{process_name}'", file=STDOUT_LIGHTCYAN)
            continue

        if p_abbreviation in processes:
            print(
                f"Registered process '{process_name}' with abbreviation '{p_abbreviation}'"
                f" (overwriting previous process)",
                file=STDOUT_CYAN
            )
        else:
            print(f"Registered process '{process_name}' with abbreviation '{p_abbreviation}'", file=STDOUT_LIGHTCYAN)

        processes[p_abbreviation] = processes[process_name]


running: bool = True
stopping: dict[str, bool] = {}
force_to_stop: bool = False


@Command(
    "q",
    description="Quits the program",
    usage="%"
          "\n├ [-e 'Use end command to terminate program instead of trying to kill process]"
          "\n└ [--t 'Timeout in seconds to wait for process to terminate (default 5, range 0 to 15 minutes)]"
)
def _quit(_cmd, ca_t: float = 5, cf_e: bool = False, *_):
    global running, stopping

    timeout = float(ca_t)
    if timeout <= 0:
        print(f"Timeout must be positive", file=STDOUT_LIGHTYELLOW)
        return
    if timeout > 60 * 15:
        print(f"Timeout must be less than 15 minutes", file=STDOUT_LIGHTYELLOW)
        return

    join_threads: list[threading.Thread] = []

    def _join(p: SubprocessService):
        try:
            p.join(timeout)
            print(f"Process '{p.name}' terminated", file=STDOUT_LIGHTGREEN)
        except TimeoutError:
            print(f"Process '{p.name}' did not terminate within {timeout} seconds", file=STDOUT_LIGHTYELLOW)

    for process in {processes[x.name] for x in processes.values()}:
        if not process.running:
            continue
        if cf_e:
            print(f"Ending process '{process.name}' ({process.end_cmd})", file=STDOUT_LIGHTBLUE)
            process.end()
        else:
            print(f"Terminating process '{process.name}'", file=STDOUT_LIGHTBLUE)
            process.stop()

        t = threading.Thread(target=_join, args=(process,), daemon=True)
        t.start()
        join_threads.append(t)

    def _join_all():
        global stopping
        for th in join_threads:
            th.join()
        print("All processes terminated", file=STDOUT_LIGHTGREEN)
        stopping["cmd.q.join_all"] = False

    stopping["cmd.q.join_all"] = True

    threading.Thread(target=_join_all, daemon=True).start()
    running = False


def _validate_processes_name(
        names: list[str],
        checker: Callable[[SubprocessService], bool | None] = lambda _: False,
        *,
        default_to_all: bool = False
) -> Generator[SubprocessService, Any, None]:
    """
    **顺序可能丢失**
    """
    name_set: set[str] = set(names)
    if (not name_set) and default_to_all:
        name_set |= {x.name for x in processes.values()}
    if '*' in name_set:
        name_set |= {x.name for x in processes.values()}
        name_set.remove('*')

    if not name_set:
        print("No processes specified", file=STDOUT_YELLOW)

    for name in name_set:
        if name not in processes:
            print(f"Process '{name}' not found", file=STDOUT_LIGHTYELLOW)
            continue

        pobj = processes[name]
        if pobj.name != name and pobj.name in name_set:
            continue
        if checker(pobj):
            continue

        yield pobj


@Command("s", description="Starts the specified process", usage="% <'*' | process name> ...")
def _start(cmd: list[str], *_):
    def checker(pobj: SubprocessService):
        if pobj.running:
            print(f"Process '{pobj.name}' is already running", file=STDOUT_YELLOW)
            return True

    for p in _validate_processes_name(cmd, checker):
        print(f"Starting process '{p.name}'", file=STDOUT_LIGHTGREEN)
        p.start()


@Command("e", description="Ends the specified process", usage="% <'*' | process name> ...")
def _end(cmd: list[str], *_):
    def checker(pobj: SubprocessService):
        if not pobj.running:
            print(f"Process '{pobj.name}' is not running", file=STDOUT_YELLOW)
            return True

    for p in _validate_processes_name(cmd, checker):
        print(f"Sending end command: '{p.end_cmd}'", file=STDOUT_MAGENTA)
        p.end()


@Command("k", description="Kills the specified process", usage="% <'*' | process name> ...")
def _kill(cmd: list[str], *_):
    def checker(pobj: SubprocessService):
        if not pobj.running:
            print(f"Process '{pobj.name}' is not running", file=STDOUT_YELLOW)
            return True

    for p in _validate_processes_name(cmd, checker):
        print(f"Killing process '{p.name}'", file=STDOUT_LIGHTGREEN)
        p.stop()


def _build_table(*titles):
    column_max_len: list[int] = [len(t) for t in titles]
    lines: list[tuple[str, ...]] = [tuple(t for t in titles)]

    def _add_line(*items: str) -> None:
        nonlocal lines, column_max_len
        for i, item in enumerate(items):
            column_max_len[i] = max(column_max_len[i], len(item))

        lines.append(items)

    def _process_line(
            line: tuple[str, ...],
            item_processor: Callable[[str, int], str],
            line_processor: Callable[[list[str]], list[str]],
    ):
        nonlocal column_max_len
        split_lines: list[list[str]] = []
        for item in line:
            item_lines: list[str] = []
            for sub_item in item.split("\n"):
                item_lines.append(sub_item)
            split_lines.append(item_lines)

        for raw_line in zip_longest(*split_lines, fillvalue=""):
            result_line: list[str] = []
            for item, max_len in zip(raw_line, column_max_len, strict=True):
                result_line.append(item_processor(item, max_len))

            result_line = line_processor(result_line)

            yield tuple(item for item in result_line)

    def _get_lines(
            item_processor: Callable[[str, int], str] = None,
            line_processor: Callable[[list[str]], list[str]] = None,
    ):
        nonlocal lines

        def _default_item_processor(item: str, max_len: int) -> str:
            return item.ljust(max_len)

        def _default_line_processor(raw_line: list[str]) -> list[str]:
            raw_line[-1] = raw_line[-1].rstrip()
            return raw_line

        item_processor = item_processor or _default_item_processor
        line_processor = line_processor or _default_line_processor

        for line in lines:
            yield from _process_line(line, item_processor, line_processor)

    return _add_line, _get_lines


TABLE_SEP = "    "
TABLE_INDENT = ' '


def _show_table(line_gen, file=None):
    if file is None:
        file = STDOUT_LIGHTMAGENTA
    print(*next(line_gen), sep=TABLE_SEP, file=file)
    for line in line_gen:
        print(end=TABLE_INDENT)
        print(*line, sep=TABLE_SEP, file=file)


def _print_wrapper(txt, file: str = None, encoding: str = "utf-8"):
    if file is not None:
        file = open(file, mode='a', encoding=encoding)
    try:
        print(txt, end='', file=file)
    finally:
        if file is not None:
            file.close()


ProcessPipeCallbacks: WeakKeyDictionary[SubprocessService, dict[str | None, Callable]] = WeakKeyDictionary()


def _pipe_name(pipe: str | None):
    return "Console" if pipe is None else pipe


@Command(
    "cp",
    description="Connects the pipe to the specified process",
    usage="% <'*' | <process name> ...\n"
          "└ [-f 'File to connect(Console if not set)]"
)
def _connect_pipe(cmd: list[str], ca_f: str = None):
    callback = functools.partial(_print_wrapper, file=ca_f)

    def checker(pobj: SubprocessService):
        if pobj not in ProcessPipeCallbacks:
            return
        if ca_f in ProcessPipeCallbacks[pobj]:
            print(f"Process '{pobj.name}' already connected to '{_pipe_name(ca_f)}'", file=STDOUT_YELLOW)
            return True

    for p in _validate_processes_name(cmd, checker):
        if p not in ProcessPipeCallbacks:
            ProcessPipeCallbacks[p] = {}

        print(f"Connecting process '{p.name}' to '{_pipe_name(ca_f)}'", file=STDOUT_LIGHTGREEN)
        p.connectStdout(callback)
        ProcessPipeCallbacks[p][ca_f] = callback


@Command(
    "dp",
    description="Disconnects the pipe from the specified process",
    usage="% <'*' | <process name> ...\n"
          "└ [-f 'File to connect(Console if not set)]"
)
def _disconnect_pipe(cmd: list[str], ca_f: str = None):
    def checker(pobj: SubprocessService):
        if ca_f not in ProcessPipeCallbacks.get(pobj, []):
            print(f"Process '{pobj.name}' is not connected to '{_pipe_name(ca_f)}'", file=STDOUT_YELLOW)
            return True

    for p in _validate_processes_name(cmd, checker):
        print(f"Disconnecting process '{p.name}' from '{_pipe_name(ca_f)}'", file=STDOUT_LIGHTGREEN)
        p.disconnectStdout(ProcessPipeCallbacks[p][ca_f])
        if p in ProcessPipeCallbacks:
            del ProcessPipeCallbacks[p][ca_f]
            if not ProcessPipeCallbacks[p]:
                del ProcessPipeCallbacks[p]


@Command(
    "rp",
    description="Checks if the pipe is registered for the specified process",
    usage="% [process name] ..."
)
def _registered_pipe(cmd: list[str], *_):
    add_line, get_lines = _build_table("Process", "Description", "Registered")

    p = None
    for p in _validate_processes_name(cmd, default_to_all=True):
        running_status: list[str] = []
        if p not in ProcessPipeCallbacks:
            running_status.append("Not registered")
        else:
            for file_name in ProcessPipeCallbacks[p]:
                if file_name is None:
                    running_status.insert(0, "Console")
                    continue
                running_status.append(f"File: '{file_name}'")

        desc = p.description or ''
        for state in running_status:
            add_line(p.name, desc, state)
            process_name, desc = '', ''

    if p is None:
        return

    _show_table(get_lines())


@Command(
    "st",
    description="Sends text to the specified process",
    usage="% <process name> ..."
          "\n├ [--t 'Text to send]"
          "\n└ [-nf 'No flush]"
)
def _send_text(cmd: list[str], ca_t: str = '', cf_nf: bool = False, *_):
    def checker(pobj: SubprocessService):
        if not pobj.running:
            print(f"Process '{pobj.name}' is not running", file=STDOUT_LIGHTYELLOW)
            return True

    for p in _validate_processes_name(cmd, checker):
        print(f"Sending text '{ca_t}' to '{p.name}'", file=STDOUT_LIGHTGREEN)
        p.sendStdin(f"{ca_t}\n\r", not cf_nf)


@Command("ps", description="Displays the status of the specified process", usage="% [process name] ...")
def _print_status(cmd: list[str], *_):
    add_line, get_lines = _build_table("Process", "Alias", "Description", "Running")

    p = None
    for p in _validate_processes_name(cmd, default_to_all=True):
        running_state = "Running" if p.running else "Stopped"
        add_line(p.name, p.abbreviation or '', p.description or '', running_state)

    if p is None:
        return

    _show_table(get_lines())


@Command(
    "as",
    description="Sets the auto-start status of the specified process",
    usage="%"
          "\n├ [-l 'List auto-start processes]"
          "\n├ [-s 'Set auto-start]"
          "\n└ [-c 'Cancel auto-start]"
)
def _auto_start(cmd: list[str], cf_l: bool = False, cf_s: bool = False, cf_c: bool = False):
    flags: dict[str, bool] = {"-l": cf_l, "-s": cf_s, "-c": cf_c}
    flag_count = sum([x for x in flags.values()])
    if flag_count > 1:
        flags_string = ', '.join(filter(lambda x: flags[x], flags.keys()))
        raise ArgumentParsingError(flags_string, f"Only one of {', '.join(flags.keys())} can be used at a time")
    elif flag_count == 0:
        print("Use '? as' for help", file=STDOUT_LIGHTYELLOW)
        return

    def _show_list():
        add_line, get_lines = _build_table("Process", "Description", "Enable AutoStart")

        p = None
        for p in _validate_processes_name(cmd, default_to_all=True):
            p_autostart = requireConfig(
                '', "process.yaml",
                {f"Register.{p.name}.auto_start": None | bool}
            ).checkConfig(ignore_missing=True).get(f"Register.{p.name}.auto_start")
            if p_autostart is None:
                p_autostart = "Not defined!"
            elif isinstance(p_autostart, bool):
                p_autostart = "Enabled" if p_autostart else "Disabled"
            else:
                p_autostart = "Invalid value!"
            add_line(p.name, p.description, p_autostart if p_autostart is not None else "Not defined!")

        if p is None:
            return

        _show_table(get_lines())

    def _set():
        for p in _validate_processes_name(cmd):
            DefaultConfigPool.get('', "process.yaml").data.setPathValue(f"Register.{p.name}.auto_start", True)
            print(f"Process {p.name} enabled auto-start", file=STDOUT_LIGHTGREEN)

    def _cancel():
        for p in _validate_processes_name(cmd):
            DefaultConfigPool.get('', "process.yaml").data.setPathValue(f"Register.{p.name}.auto_start", False)
            print(f"Process {p.name} disabled auto-start", file=STDOUT_LIGHTGREEN)

    if cf_l:
        _show_list()
    elif cf_s:
        _set()
    elif cf_c:
        _cancel()


@Command("?", description="Displays the description and usage of the specified command", usage="% [command] ...")
def _help(cmd_ls: list[str], *_):
    if not cmd_ls:
        command_list: dict = DefaultCommandList.data
    else:
        command_list = {}
        for c in set(cmd_ls):
            try:
                command_list |= {c: DefaultCommandList[c]}
            except KeyError:
                print(f"Command {c} not found", file=STDOUT_LIGHTYELLOW)

    if not command_list:
        return

    add_line, get_lines = _build_table("Command", "Description", "Usage")

    for cmd, data in command_list.items():
        desc = data["description"]
        usage = data["usage"]

        add_line(cmd, desc, usage)

    _show_table(get_lines())


@Command("sc", description="Saves the current config to the config file", usage="%")
def _save_config(*_):
    DefaultConfigPool.saveAll()
    print("Config saved", file=STDOUT_LIGHTGREEN)


@Command("db",
         description="Debug command",
         usage="%"
               "\n├ [-c 'Show colors]"
               "\n└ ['Any flag or argument] ...")
def _debug(*args, **kwargs):
    print(args, kwargs, file=STDOUT_LIGHTMAGENTA)
    if kwargs.get("cf_c"):
        colors = {
            "Red": STDOUT_RED,
            "Green": STDOUT_GREEN,
            "Yellow": STDOUT_YELLOW,
            "Blue": STDOUT_BLUE,
            "Magenta": STDOUT_MAGENTA,
            "Cyan": STDOUT_CYAN,

            "Light Red": STDOUT_LIGHTRED,
            "Light Green": STDOUT_LIGHTGREEN,
            "Light Yellow": STDOUT_LIGHTYELLOW,
            "Light Blue": STDOUT_LIGHTBLUE,
            "Light Magenta": STDOUT_LIGHTMAGENTA,
            "Light Cyan": STDOUT_LIGHTCYAN
        }

        for color, file in colors.items():
            print(color, file=file)

    if kwargs.get("ca_eval"):
        for cmd in kwargs["ca_eval"]:
            print(eval(cmd))

    if kwargs.get("ca_exec"):
        for cmd in kwargs["ca_exec"]:
            print(exec(cmd))


class ArgumentParsingError(CommandException):
    def __init__(self, argument: str, message: str):
        self.argument = argument
        self.message = message

        super().__init__(f"Error parsing argument '{argument}': {message}")


def _rc_args_maker(string: str | Any, *_, **__):
    if type(string) is not str:
        return string

    raw_ls: list[str | None] = string.split(' ')
    flags: list[str] = []
    arguments: dict[str, list[str]] = {}

    parsing_argument: str | None = None
    argument_cache: str | None = None

    escape_at_end: bool = False

    def check_string(s: str) -> str:
        nonlocal escape_at_end
        result: str = ''
        escape_map = {
            '"': '"',
            '\\': '\\',
            'n': '\n',
            'r': '\r',
            't': '\t'
        }
        length = len(s)
        index = 0
        while index < length:
            char = s[index]
            index += 1

            if (char != '\\') and (not escape_at_end):
                if char == '"':
                    raise ArgumentParsingError(
                        f"--{parsing_argument}",
                        f"Found unexpected data after argument: '{s[index:]}'"
                    )
                result += char
                continue

            if escape_at_end:
                index -= 1

            if index >= length:
                escape_at_end = True
                break

            escape_at_end = False
            try:
                result += escape_map[s[index]]
            except KeyError as e:
                raise ArgumentParsingError(  # 转义符未找到
                    f"--{parsing_argument}",
                    f"Invalid escape: '\\{s[index]}'"
                ) from e
            index += 1

        return result

    def _parse_argument(now_item: str):
        nonlocal arguments, parsing_argument, argument_cache

        def _count_end(value: str):
            count: int = 0
            while value.endswith('\\'):
                value = value[:-1]
                count += 1
            return count

        is_first = argument_cache is None
        if is_first and (not now_item.startswith('"')):
            arguments[parsing_argument][-1] = now_item
            parsing_argument = None
            return

        is_end = now_item.endswith('"') and (_count_end(now_item[:-1]) % 2 == 0)
        if is_first:
            if is_end and now_item != '"':
                arguments[parsing_argument][-1] = check_string(now_item[1:-1])
                parsing_argument = None
                return
            argument_cache = check_string(now_item[1:])
            return

        if is_end:
            argument_cache += check_string(f" {now_item[:-1]}")
            arguments[parsing_argument][-1] = argument_cache
            parsing_argument = None
            return

        argument_cache += check_string(f" {now_item}")

    for i in range(len(raw_ls)):
        item = raw_ls[i]

        if parsing_argument is not None:
            _parse_argument(item)
            raw_ls[i] = None
            continue

        if item[:1] != '-':
            continue

        if item[1:2] != '-':
            flags.append(item[1:])
            raw_ls[i] = None
            continue

        raw_ls[i] = None
        if len(raw_ls) <= (i + 1):
            raise ArgumentParsingError(item, "No value provided")

        parsing_argument = item[2:]
        if parsing_argument not in arguments:
            arguments[parsing_argument] = []
        arguments[parsing_argument].append('')
        raw_ls[i] = None

    if parsing_argument is not None:
        raise ArgumentParsingError(
            f"--{parsing_argument}",
            f"Unexpected end of argument: '\"{argument_cache}'"
        )

    return [x for x in raw_ls if (x is not None) and (x != '')], arguments, flags


def _rc_args_unpacker(*args, func, **kwargs):
    (cmd, cmd_args, cmd_flags), *args = args
    cmd: list[str]
    cmd_args: dict[str, str]
    cmd_flags: list[str | list[str]]

    full_arg_spec = inspect.getfullargspec(func)
    annotations = full_arg_spec.annotations
    default_index = len(full_arg_spec.args) - len(full_arg_spec.defaults if full_arg_spec.defaults is not None else [])

    cmd_multi_args_require: set[str] = {n[4:] for n in full_arg_spec.args if n.startswith("cma_")}
    cmd_args_require: set[str] = {n[3:] for n in full_arg_spec.args if n.startswith("ca_")} | cmd_multi_args_require
    cmd_flags_require: set[str] = {n[3:] for n in full_arg_spec.args if n.startswith("cf_")}
    if full_arg_spec.varkw is not None:
        cmd_multi_args_require |= cmd_args.keys()
        cmd_args_require |= cmd_args.keys()
        cmd_flags_require |= set(cmd_flags)

    required_kwargs: dict[str, ...] = {}
    for arg in cmd_args_require:

        if not (arg in cmd_args):
            has_default = default_index <= full_arg_spec.args.index(f"ca_{arg}")
            if has_default:
                continue
            raise ArgumentParsingError(f"--{arg}", "Required argument not provided")

        if isinstance(cmd_args[arg], list) and (arg not in cmd_multi_args_require):
            arg_length = len(cmd_args[arg])
            if arg_length > 1:
                raise ArgumentParsingError(f"--{arg}", "Argument more than one")
            if arg_length < 1:
                raise ArgumentParsingError(f"--{arg}", "Argument less than one")

            cmd_args[arg] = cmd_args[arg][0]

        if f"ca_{arg}" in annotations:
            try:
                cmd_args[arg] = annotations[f"ca_{arg}"](cmd_args[arg])
            except (ValueError, TypeError) as e:
                raise ArgumentParsingError(arg, str(e)) from e

        required_kwargs[f"ca_{arg}"] = cmd_args[arg]
        del cmd_args[arg]

    for flag in cmd_flags_require:

        if not (flag in cmd_flags):
            has_default = default_index <= full_arg_spec.args.index(f"cf_{flag}")
            if has_default:
                continue

            raise ArgumentParsingError(f"-{flag}", "Required flag not provided")

        required_kwargs[f"cf_{flag}"] = True
        while flag in cmd_flags:
            cmd_flags.remove(flag)

    for arg in cmd_args:
        raise ArgumentParsingError(f"--{arg}", "Unexpected argument provided")

    for flag in cmd_flags:
        raise ArgumentParsingError(f"-{flag}", "Unexpected flag provided")

    return func(cmd[1:], *args, **required_kwargs, **kwargs)


def main():
    start_processes()
    run_command = RunCommand(args_maker=_rc_args_maker, args_unpacker=_rc_args_unpacker)

    for pobj in {processes[x.name] for x in processes.values()}:
        if not pobj.process_config.get("auto_start"):
            continue
        try:
            _start([pobj.name])
        except Exception as e:
            print(f"Failed to start {pobj.name}: {e}", file=STDOUT_LIGHTRED)

    while running:
        input_str = input()

        if not input_str:
            continue

        try:
            run_command.run_by_str(f"${input_str}", float("inf"))
        except CommandException as e:
            print(f"{type(e).__name__}: {e}", file=sys.stderr)
        except Exception as e:
            traceback.print_exception(e)
            print(f"An unexpected error occurred: {type(e).__name__}: {e}", file=sys.stderr)

    while any(stopping.values()):
        print(f"\nWaiting for {list(stopping.keys())} to stop...", file=STDOUT_BLUE)
        if force_to_stop:
            print("Force stopping...", file=STDOUT_BLUE)
            break
        time.sleep(0.5)


if __name__ == "__main__":
    requireConfig('', "process.yaml", default_config["process"]).checkConfig()
    DefaultConfigPool.saveAll()
    main()

__all__ = ("main",)
