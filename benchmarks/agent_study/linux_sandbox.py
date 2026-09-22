"""Linux worker filesystem/network confinement; not an in-process anti-tamper proof."""
import ctypes
import os
from pathlib import Path
import platform
import sys


def confine(writable, readonly):
    if sys.platform != "linux" or platform.machine() != "x86_64":
        raise RuntimeError("confinement requires Linux x86_64")
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    sec = ctypes.CDLL("libseccomp.so.2")
    abi = libc.syscall(444, 0, 0, 1)
    if abi < 3:
        raise RuntimeError("Landlock ABI >= 3 required")

    class Ruleset(ctypes.Structure):
        _fields_ = [("handled_access_fs", ctypes.c_uint64)]

    class PathRule(ctypes.Structure):
        _pack_ = 1
        _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]

    rights = (1 << 15) - 1
    rules = Ruleset(rights)
    fd = libc.syscall(444, ctypes.byref(rules), ctypes.sizeof(rules), 0)
    if fd < 0:
        raise OSError(ctypes.get_errno(), "landlock_create_ruleset")
    try:
        paths = [(Path(p), 4 | 8) for p in readonly if Path(p).exists()]
        paths.append((Path(writable), rights & ~1))
        for path, access in paths:
            path = path.resolve()
            handle = os.open(path, os.O_PATH | os.O_CLOEXEC)
            try:
                if not path.is_dir():
                    access &= 4 | 2 | (1 << 14)
                rule = PathRule(access, handle)
                if libc.syscall(445, fd, 1, ctypes.byref(rule), 0) < 0:
                    raise OSError(ctypes.get_errno(), f"landlock_add_rule {path}")
            finally:
                os.close(handle)
        if libc.prctl(38, 1, 0, 0, 0) or libc.syscall(446, fd, 0):
            raise OSError(ctypes.get_errno(), "landlock_restrict_self")
    finally:
        os.close(fd)
    sec.seccomp_init.argtypes = [ctypes.c_uint32]
    sec.seccomp_init.restype = ctypes.c_void_p
    sec.seccomp_rule_add.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint]
    sec.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    sec.seccomp_load.argtypes = [ctypes.c_void_p]
    sec.seccomp_release.argtypes = [ctypes.c_void_p]
    context = sec.seccomp_init(0x7fff0000)
    if not context:
        raise RuntimeError("seccomp initialization failed")
    try:
        for name in ("socket", "socketpair", "connect", "bind", "listen", "accept", "accept4",
                     "execve", "execveat", "fork", "vfork", "clone", "clone3", "ptrace",
                     "process_vm_readv", "process_vm_writev", "kill", "tkill", "tgkill",
                     "bpf", "io_uring_setup", "unshare", "setns", "mount", "umount2"):
            number = sec.seccomp_syscall_resolve_name(name.encode())
            if number >= 0 and sec.seccomp_rule_add(context, 0x00050000 | 1, number, 0):
                raise RuntimeError(f"seccomp rule failed: {name}")
        if sec.seccomp_load(context):
            raise RuntimeError("seccomp load failed")
    finally:
        sec.seccomp_release(context)
    return {"landlock_abi": int(abi), "network": "socket syscalls denied",
            "writes": str(writable), "subprocesses": "denied"}
