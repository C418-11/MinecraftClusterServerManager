# -*- coding: utf-8 -*-
# cython: language_level = 3

__author__ = "C418____11 <553515788@qq.com>"
__version__ = "0.0.1Dev"

import codecs
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

import colorama

from StdColor import ColorWrite
from buffer import StringBuffer
from command_tools import Command
from command_tools import CommandException
from command_tools import DefaultCommandList
from command_tools import RunCommand
from config import ConfigData
from config import DefaultConfigPool
from config import YamlSL
from config import requireConfig

YamlSL.enable()
YamlSL().registerTo()

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
        self._process_config: ConfigData = deepcopy(process_config)

        del self._process_config["cmd"]
        del self._process_config["workdir"]
        if self._abbreviation is not None:
            del self._process_config["abbreviation"]
        if self._description is not None:
            del self._process_config["description"]
        if self._end_cmd is not None:
            del self._process_config["end"]
        if not os.path.exists(self._workdir):
            raise FileNotFoundError(f"Workdir {self._workdir} not found")

        self._running: bool = False
        self._process: subprocess.Popen | None = None

        self._stdout_buffer: StringBuffer = StringBuffer()
        self._thread: threading.Thread | None = None

    def _buffer_loop(self):
        cache: bytes = b''
        while self._running and self._process.poll() is None:
            cache += self._process.stdout.read(1)

            try:
                txt = cache.decode(self._stdout_encoding)
                cache = b''
            except UnicodeDecodeError as e:
                if len(cache) > 5:
                    txt = '?'
                    cache = cache[:e.start] + cache[e.end:]
                else:
                    continue

            self._stdout_buffer.write(txt)

        if self._process.poll() is not None:
            self._running = False
            self._process = None

    def connectStdout(self, callback: Callable[[str], None]) -> None:
        self._stdout_buffer.register(callback)

    def disconnectStdout(self, callback: Callable[[str], None]) -> None:
        self._stdout_buffer.unregister(callback)

    def isConnectedStdout(self, callback: Callable[[str], None]) -> bool:
        return self._stdout_buffer.isRegistered(callback)

    def sendStdin(self, txt: str) -> None:
        if self._process.poll() is not None:
            raise RuntimeError("Process not running")

        self._process.stdin.write(txt.encode(self._stdin_encoding))
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
        self._running = False

    def end(self):
        if not self._running:
            raise RuntimeError("Process not running")

        if self._end_cmd is None or self._end_cmd == "^C":
            self._process.terminate()
        else:
            print(f"Sending end command: {self._end_cmd}", file=STDOUT_MAGENTA)
            self.sendStdin(f"{self._end_cmd}\n")

    def join(self, timeout: float | None = None):
        if self._process is not None:
            try:
                self._process.wait(timeout)
            except subprocess.TimeoutExpired as e:
                raise TimeoutError(f"Process {self._name} did not end within {timeout} seconds") from e

        if self._thread is not None:
            self._thread.join(timeout)
            if self._thread.is_alive():
                raise TimeoutError(f"Thread {self._thread.name} did not join within {timeout} seconds")

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
            print(f"Process {p.name} terminated", file=STDOUT_LIGHTGREEN)
        except TimeoutError:
            print(f"Process {p.name} did not terminate within {timeout} seconds", file=STDOUT_LIGHTYELLOW)

    for process in processes.values():
        if not process.running:
            print(f"Process {process.name} is not running", file=STDOUT_YELLOW)
            continue
        if cf_e:
            print(f"Ending process {process.name}", file=STDOUT_LIGHTBLUE)
            process.end()
        else:
            print(f"Terminating process {process.name}", file=STDOUT_LIGHTBLUE)
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


@Command("s", description="Starts the specified process", usage="% <'*' | process name> ...")
def _start(cmd: list[str], *_):
    if '*' in cmd:
        cmd = processes.keys()

    for name in cmd:
        p = processes.get(name)
        if p is None:
            print(f"Process {name} not found", file=STDOUT_LIGHTYELLOW)
            continue

        if p.running:
            print(f"Process {name} is already running", file=STDOUT_YELLOW)
            continue
        p.start()
        print(f"Started process {name}", file=STDOUT_LIGHTGREEN)


@Command("e", description="Ends the specified process", usage="% <'*' | process name> ...")
def _end(cmd: list[str], *_):
    if '*' in cmd:
        cmd = processes.keys()

    for name in cmd:
        p = processes.get(name)
        if p is None:
            print(f"Process {name} not found", file=STDOUT_LIGHTYELLOW)
            continue

        if not p.running:
            print(f"Process {name} is not running", file=STDOUT_YELLOW)
            continue
        p.end()
        print(f"Ended process {name}", file=STDOUT_LIGHTGREEN)


@Command("k", description="Kills the specified process", usage="% <'*' | process name> ...")
def _kill(cmd: list[str], *_):
    if '*' in cmd:
        cmd = processes.keys()

    for name in cmd:
        p = processes.get(name)
        if p is None:
            print(f"Process {name} not found", file=STDOUT_LIGHTYELLOW)
            continue

        if not p.running:
            print(f"Process {name} is not running", file=STDOUT_YELLOW)
            continue
        p.stop()
        print(f"Killed process {name}", file=STDOUT_LIGHTGREEN)


def _print_wrapper(txt):
    print(txt, end="")


def _build_list(*titles):
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


@Command("cp", description="Connects the pipe to the specified process", usage="% <'*' | <process name> ...")
def _connect_pipe(cmd: list[str], *_):
    if '*' in cmd:
        cmd = processes.keys()

    for name in cmd:
        p = processes.get(name)
        if p is None:
            print(f"Process {name} not found", file=STDOUT_LIGHTYELLOW)
            continue

        if p.isConnectedStdout(_print_wrapper):
            print(f"Pipe for process {name} is already connected", file=STDOUT_YELLOW)
            continue

        print(f"Connected pipe for process {name}", file=STDOUT_LIGHTGREEN)
        p.connectStdout(_print_wrapper)


@Command("dp", description="Disconnects the pipe from the specified process", usage="% <'*' | <process name> ...")
def _disconnect_pipe(cmd: list[str], *_):
    if '*' in cmd:
        cmd = processes.keys()

    for name in cmd:
        p = processes.get(name)
        if p is None:
            print(f"Process {name} not found", file=STDOUT_LIGHTYELLOW)
            continue

        if not p.isConnectedStdout(_print_wrapper):
            print(f"Pipe for process {name} is not connected", file=STDOUT_YELLOW)
            continue

        print(f"Disconnected pipe for process {name}", file=STDOUT_LIGHTGREEN)
        p.disconnectStdout(_print_wrapper)


@Command(
    "rp",
    description="Checks if the pipe is registered for the specified process",
    usage="% [process name] ..."
)
def _registered_pipe(cmd_ls: list[str], *_):
    cmd: set[str] = set(cmd_ls)
    if not cmd:
        cmd = set(processes.keys())

    add_line, get_lines = _build_list("Process", "Description", "Registered")

    for process_name in cmd.copy():
        p = processes.get(process_name)
        if p is None:
            print(f"Process {process_name} not found", file=STDOUT_LIGHTYELLOW)
            cmd.remove(process_name)
            continue

        running_state = "Registered" if p.isConnectedStdout(_print_wrapper) else "Not registered"
        add_line(process_name, p.description or "", running_state)

    if not cmd:
        return

    line_gen = get_lines()
    print(*next(line_gen), sep="     ", file=STDOUT_LIGHTMAGENTA)
    for line in line_gen:
        print(end=' ')
        print(*line, sep="     ", file=STDOUT_LIGHTMAGENTA)


@Command(
    "st",
    description="Sends text to the specified process",
    usage="% <process name> ..."
          "\n└ [--t 'Text to send]"
)
def _send_text(cmd: list[str], ca_t: str = '', *_):
    process_to_send = []
    for name in set(cmd):
        p = processes.get(name)
        if p is None:
            print(f"Process {name} not found", file=STDOUT_LIGHTYELLOW)
            continue

        process_to_send.append(p)

    for p in process_to_send:
        if not p.running:
            print(f"Process {p.name} is not running", file=STDOUT_LIGHTYELLOW)
            continue

        print(f"Sent to process {p.name}: {ca_t}", file=STDOUT_LIGHTGREEN)
        p.sendStdin(f"{ca_t}\n")


@Command("ps", description="Displays the status of the specified process", usage="% [process name] ...")
def _print_status(cmd_ls: list[str], *_):
    cmd: set[str] = set(cmd_ls)
    if not cmd:
        cmd = set(processes.keys())

    add_line, get_lines = _build_list("Process", "Description", "Running")

    for name in cmd.copy():
        p = processes.get(name)
        if p is None:
            print(f"Process {name} not found", file=STDOUT_LIGHTYELLOW)
            cmd.remove(name)
            continue

        running_state = "Running" if p.running else "Stopped"
        add_line(name, p.description or "", running_state)

    if not cmd:
        return

    line_gen = get_lines()
    print(*next(line_gen), sep="     ", file=STDOUT_LIGHTMAGENTA)
    for line in line_gen:
        print(end=' ')
        print(*line, sep="     ", file=STDOUT_LIGHTMAGENTA)


@Command(
    "as",
    description="TODO",
    usage="%"
          "\n├ [-ls 'List all]"
          "\n├ [--a 'Add]"
          "\n└ [--r 'Remove]"
)
def _auto_start(_, cf_ls: bool = False, ca_a: str = '', ca_r: str = ''):  # TODO
    print("TODO", file=STDOUT_LIGHTYELLOW)
    print(cf_ls, ca_a, ca_r)
    pass


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

    add_line, get_lines = _build_list("Command", "Description", "Usage")

    for cmd, data in command_list.items():
        desc = data["description"]
        usage = data["usage"]

        add_line(cmd, desc, usage)

    line_gen = get_lines()
    print("     ".join(next(line_gen)), file=STDOUT_LIGHTMAGENTA)
    for line in line_gen:
        print(end=' ')
        print("     ".join(line), file=STDOUT_LIGHTMAGENTA)


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
    arguments: dict[str, str] = {}

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

    def _parse_argument(now_item):
        nonlocal arguments, parsing_argument, argument_cache

        def _count_end(value: str):
            count: int = 0
            while value.endswith('\\'):
                value = value[:-1]
                count += 1
            return count

        is_first = argument_cache is None
        if is_first and (not now_item.startswith('"')):
            arguments[parsing_argument] = now_item
            parsing_argument = None
            return

        is_end = now_item.endswith('"') and (_count_end(now_item[:-1]) % 2 == 0)
        if is_first:
            if is_end and now_item != '"':
                arguments[parsing_argument] = check_string(now_item[1:-1])
                parsing_argument = None
                return
            argument_cache = check_string(now_item[1:])
            return

        if is_end:
            argument_cache += check_string(f" {now_item[:-1]}")
            arguments[parsing_argument] = argument_cache
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
    cmd_flags: list[str]

    full_arg_spec = inspect.getfullargspec(func)
    annotations = full_arg_spec.annotations
    default_index = len(full_arg_spec.args) - len(full_arg_spec.defaults if full_arg_spec.defaults is not None else [])

    cmd_args_require: set[str] = {n[3:] for n in full_arg_spec.args if n.startswith("ca_")}
    cmd_flags_require: set[str] = {n[3:] for n in full_arg_spec.args if n.startswith("cf_")}
    if full_arg_spec.varkw is not None:
        cmd_args_require |= cmd_args.keys()
        cmd_flags_require |= set(cmd_flags)

    required_kwargs: dict[str, ...] = {}
    for arg in cmd_args_require:

        if not (arg in cmd_args):
            has_default = default_index <= full_arg_spec.args.index(f"ca_{arg}")
            if has_default:
                continue
            raise ArgumentParsingError(f"--{arg}", "Required argument not provided")

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

    while running:
        input_str = input()

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
