# -*- coding: utf-8 -*-
# cython: language_level = 3

__author__ = "C418____11 <553515788@qq.com>"
__version__ = "0.0.1Dev"

import codecs
import subprocess
import sys
import threading
from copy import deepcopy
from platform import system
from typing import Callable

from buffer import StringBuffer
from command_tools import Command
from command_tools import CommandNotFindError
from command_tools import DefaultCommandList
from command_tools import RunCommand
from config import ConfigData
from config import DefaultConfigPool
from config import requireConfig

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
                "end": "exit"
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

        self._abbreviation: str | None = process_config.get("abbreviation")
        self._description: str | None = process_config.get("description")
        self._stdin_encoding: str = process_config.get(
            "Encoding.stdin", default_encoding.get("stdin", "utf-8")
        )
        self._stdout_encoding: str = process_config.get(
            "Encoding.stdout", default_encoding.get("stdout", "utf-8")
        )
        self._start_cmd: str | list[str] = process_config["cmd"]
        self._end_cmd: str | None = process_config.get("end")
        self._workdir: str = process_config["workdir"]
        self._process_config: ConfigData = deepcopy(process_config)

        del self._process_config["cmd"]
        if self._abbreviation is not None:
            del self._process_config["abbreviation"]

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

        self._running = False
        self._process = None

    def connectStdout(self, callback: Callable[[str], None]) -> None:
        self._stdout_buffer.register(callback)

    def disconnectStdout(self, callback: Callable[[str], None]) -> None:
        self._stdout_buffer.unregister(callback)

    def isConnectedStdout(self, callback: Callable[[str], None]) -> bool:
        return self._stdout_buffer.isRegistered(callback)

    def sendStdin(self, txt: str) -> None:
        if self._process is None:
            raise RuntimeError("Process not started")

        self._process.stdin.write(txt.encode(self._stdin_encoding))
        self._process.stdin.flush()

    def start(self):
        if self._process is not None:
            raise ValueError("Process already started")

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
        if self._process is None:
            raise RuntimeError("Process not started")

        self._process.terminate()
        self._process = None
        self._running = False

    def end(self):
        if self._process is None:
            raise RuntimeError("Process not started")

        if self._end_cmd is None or self._end_cmd == "^C":
            self._process.terminate()
        else:
            print(f"Sending end command: {self._end_cmd}")
            self.sendStdin(f"{self._end_cmd}\n")

    def join(self, timeout: float | None = None):
        if self._thread is None:
            return

        if self._process is not None:
            self._process.wait(timeout)
        self._thread.join(timeout)
        self._thread = None

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
        if self._process is None:
            raise RuntimeError("Process not started")

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

        processes[process_name] = SubprocessService(process_name)

        p_abbreviation = processes[process_name].abbreviation
        if p_abbreviation is None:
            print(f"Registered process '{process_name}'")
            continue

        if p_abbreviation in processes:
            print(f"Abbreviation '{p_abbreviation}' already exists, skipping registration")
            continue

        processes[p_abbreviation] = processes[process_name]
        print(f"Registered process '{process_name}' with abbreviation '{p_abbreviation}'")


running: bool = True


@Command("q", description="Quits the program", usage="%")
def _exit(*_):
    global running
    running = False
    for process in processes.values():
        if not process.running:
            continue
        print(f"Terminating process {process.pid}")
        process.stop()
        process.join(5)
        if process.is_alive():
            print(f"Failed to terminate process {process.pid}")
    sys.exit(0)


@Command("s", description="Starts the specified process", usage="% ('*', '<process name> ...')")
def _start(cmd: list[str]):
    if '*' in cmd:
        cmd = processes.keys()

    for name in cmd:
        p = processes.get(name)
        if p is None:
            print(f"Process {name} not found")
            continue

        if p.running:
            print(f"Process {name} is already running")
            continue
        p.start()
        print(f"Started process {name}")


@Command("e", description="Ends the specified process", usage="% ('*', '<process name> ...')")
def _end(cmd: list[str]):
    if '*' in cmd:
        cmd = processes.keys()

    for name in cmd:
        p = processes.get(name)
        if p is None:
            print(f"Process {name} not found")
            continue

        if not p.running:
            print(f"Process {name} is not running")
            continue
        p.end()
        print(f"Ended process {name}")


@Command("k", description="Kills the specified process", usage="% ('*', '<process name> ...')")
def _kill(cmd: list[str]):
    if '*' in cmd:
        cmd = processes.keys()

    for name in cmd:
        p = processes.get(name)
        if p is None:
            print(f"Process {name} not found")
            continue

        if not p.running:
            print(f"Process {name} is not running")
            continue
        p.stop()
        print(f"Killed process {name}")


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

    def _get_lines():
        nonlocal lines, column_max_len
        for line in lines:
            yield (f"{item:<{column_max_len[i]}}" for i, item in enumerate(line))

    return _add_line, _get_lines


@Command("cp", description="Connects the pipe to the specified process", usage="% ('*', '<process name> ...')")
def _connect_pipe(cmd: list[str]):
    if '*' in cmd:
        cmd = processes.keys()

    for name in cmd:
        p = processes.get(name)
        if p is None:
            print(f"Process {name} not found")
            continue

        if p.isConnectedStdout(_print_wrapper):
            print(f"Pipe for process {name} is already connected")
            continue

        print(f"Connected pipe for process {name}")
        p.connectStdout(_print_wrapper)


@Command("dp", description="Disconnects the pipe from the specified process", usage="% ('*', '<process name> ...')")
def _disconnect_pipe(cmd: list[str]):
    if '*' in cmd:
        cmd = processes.keys()

    for name in cmd:
        p = processes.get(name)
        if p is None:
            print(f"Process {name} not found")
            continue

        if not p.isConnectedStdout(_print_wrapper):
            print(f"Pipe for process {name} is not connected")
            continue

        print(f"Disconnected pipe for process {name}")
        p.disconnectStdout(_print_wrapper)


@Command(
    "rp",
    description="Checks if the pipe is registered for the specified process",
    usage="% [process name] ..."
)
def _registered_pipe(cmd):
    if not cmd:
        cmd = processes.keys()

    add_line, get_lines = _build_list("Process", "Description", "Registered")

    for process_name in cmd:
        p = processes.get(process_name)
        if p is None:
            print(f"Process {process_name} not found")
            continue

        running_state = "Registered" if p.isConnectedStdout(_print_wrapper) else "Not registered"
        add_line(process_name, p.description or "", running_state)

    line_gen = get_lines()
    print(*next(line_gen), sep="     ")
    for line in line_gen:
        print(end=' ')
        print(*line, sep="     ")


@Command(
    "st",
    description="Sends text to the specified process",
    usage="% <process name> ... ('\\', '|', ',', '/') [text] ..."
)
def _send_text(cmd: list[str]):
    process_to_send = []
    for i, name in enumerate(cmd, start=1):
        if name in {'\\', '|', ',', '/'}:
            cmd = cmd[i:]
            break

        p = processes.get(name)
        if p is None:
            print(f"Process {name} not found")
            continue

        process_to_send.append(p)

    txt = ' '.join(cmd)
    for p in process_to_send:
        if not p.running:
            print(f"Process {p.name} is not running")
            continue

        print(f"Sent to process {p.name}: {txt}")
        p.sendStdin(f"{txt}\n")


@Command("ps", description="Displays the status of the specified process", usage="% [process name] ...")
def _print_status(cmd: list[str]):
    if not cmd:
        cmd = processes.keys()

    add_line, get_lines = _build_list("Process", "Description", "Running")

    for name in cmd:
        p = processes.get(name)
        if p is None:
            print(f"Process {name} not found")
            continue

        running_state = "Running" if p.running else "Stopped"
        add_line(name, p.description or "", running_state)

    line_gen = get_lines()
    print(*next(line_gen), sep="     ")
    for line in line_gen:
        print(end=' ')
        print(*line, sep="     ")


@Command("?", description="Displays the description and usage of the specified command", usage="% [command] ...")
def _help(cmd: list[str]):
    if not cmd:
        command_list: dict = DefaultCommandList.data
    else:
        command_list = {}
        for c in cmd:
            try:
                command_list |= {c: DefaultCommandList[c]}
            except KeyError:
                print(f"Command {c} not found")

    add_line, get_lines = _build_list("Command", "Description", "Usage")

    for cmd, data in command_list.items():
        desc = data["description"]
        usage = data["usage"]

        add_line(cmd, desc, usage)

    line_gen = get_lines()
    print(*next(line_gen), sep="     ")
    for line in line_gen:
        print(end=' ')
        print(*line, sep="     ")


@Command("sc", description="Saves the current config to the config file", usage="%")
def _save_config(*_):
    DefaultConfigPool.saveAll()
    print("Config saved")


def _rc_args_unpacker(*args, func, **kwargs):
    cmd, *args = args
    return func(cmd[1:], *args, **kwargs)


def main():
    start_processes()

    run_command = RunCommand(args_unpacker=_rc_args_unpacker)

    while running:
        input_str = input()

        try:
            run_command.run_by_str(f"${input_str}", float("inf"))
        except CommandNotFindError:
            print("Command not found")


if __name__ == "__main__":
    requireConfig('', "process.yaml", default_config["process"]).checkConfig()
    DefaultConfigPool.saveAll()
    main()

__all__ = ("main",)
