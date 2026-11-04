# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Mooncake 服务的进程管理与连通性工具：Ray actor 启动 master，子进程日志回流，退出时终止进程组。
# 这里只管理服务和地址，不传输 hidden states；端口连通检查也不等同一次完整对象读写验证。

import atexit
import ctypes
import os
import random
import shutil
import signal
import socket
import subprocess
import threading
import time
from urllib.parse import urlparse

import ray

from torchspec.ray.ray_actor import RayActor
from torchspec.utils.env import get_torchspec_env_vars
from torchspec.utils.logging import logger


# 依次采用构建目录环境变量、PATH 搜索、用户目录默认构建路径；只返回路径，调用者负责存在性检查。
def resolve_mooncake_master_bin() -> str:
    """Resolve the path to the mooncake_master binary."""
    if "MOONCAKE_BUILD_DIR" in os.environ:
        return os.path.join(os.environ["MOONCAKE_BUILD_DIR"], "mooncake-store/src/mooncake_master")

    which_result = shutil.which("mooncake_master")
    if which_result:
        return which_result

    home = os.path.expanduser("~")
    return os.path.join(home, "build/mooncake-store/src/mooncake_master")


# 在 Linux 子进程 exec 前建立独立进程组并设置父进程死亡信号，便于清理包装脚本及其后代。
def _subprocess_preexec():
    """Pre-exec setup for the mooncake master subprocess.

    - os.setpgrp(): Create a new process group so that os.killpg() can kill
      the wrapper script AND the real binary it spawns (grandchild).
    - PR_SET_PDEATHSIG: Kernel sends SIGTERM when the parent (Ray worker) dies,
      preventing orphans on crashes.
    """
    os.setpgrp()
    PR_SET_PDEATHSIG = 1
    ctypes.CDLL("libc.so.6").prctl(PR_SET_PDEATHSIG, signal.SIGTERM)


# Ray 持有 master 子进程的生命周期入口，让服务位置、日志和关闭操作纳入分布式编排。
class MooncakeMaster(RayActor):
    """Ray actor that wraps the mooncake master subprocess.

    Provides automatic lifecycle management — when the actor is killed or garbage
    collected, the subprocess is terminated. Logs are streamed through Ray's
    logging pipeline instead of written to files.
    """

    # 初始化空子进程和地址信息；实际启动等待显式 start 调用。
    def __init__(self):
        self._process = None
        self._info = {}

    # 拼装 gRPC、HTTP 元数据、指标端口和毫秒租约参数，启动进程并拉起日志线程，再确认未立即退出。
    def start(
        self,
        port: int,
        http_port: int,
        http_host: str = "0.0.0.0",
        kv_lease_ttl_s: float = 5.0,
    ) -> dict:
        """Launch the mooncake master subprocess.

        Args:
            port: gRPC port for mooncake master.
            http_port: HTTP metadata server port.
            http_host: HTTP metadata server host.
            kv_lease_ttl_s: Default KV object lease TTL in seconds.

        Returns:
            Dict with "master_addr" and "metadata_port".

        Raises:
            FileNotFoundError: If binary is not found.
            RuntimeError: If process fails to start.
        """
        mooncake_bin = resolve_mooncake_master_bin()
        if not os.path.exists(mooncake_bin):
            raise FileNotFoundError(f"mooncake_master binary not found at {mooncake_bin}")

        # Pick a free port for metrics / admin server
        metrics_port = self.find_free_port(start_port=random.randint(9100, 10100))
        # 租约从秒转换成原生命令行需要的毫秒；HTTP 元数据和 gRPC master 使用独立端口。
        cmd = [
            mooncake_bin,
            f"--port={port}",
            f"--http_metadata_server_port={http_port}",
            f"--http_metadata_server_host={http_host}",
            "--enable_http_metadata_server=true",
            f"--default_kv_lease_ttl={int(kv_lease_ttl_s * 1000)}",
            f"--metrics_port={metrics_port}",
        ]

        logger.info(f"Starting mooncake master on port {port}")

        # 进程组和父进程死亡信号由 preexec 设置，保证关闭 actor 时能够覆盖包装脚本派生的实际服务进程。
        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=_subprocess_preexec,
        )

        # Stream stdout/stderr through logger in background daemon threads
        self._start_log_thread(self._process.stdout, "stdout")
        self._start_log_thread(self._process.stderr, "stderr")

        time.sleep(2)

        # 这里只确认进程没有立即退出；真正客户端在模型加载前还会探测服务端口。
        if self._process.poll() is not None:
            raise RuntimeError(
                f"mooncake master failed to start (exit code: {self._process.returncode})"
            )

        host = self.get_node_ip()
        self._info = {
            "master_addr": f"{host}:{port}",
            "metadata_port": http_port,
        }

        logger.info(f"mooncake master started (PID: {self._process.pid})")
        return self._info

    # 检查子进程 poll 状态，说明进程还活着；并不探测服务实际处理能力。
    def health_check(self) -> bool:
        """Check if the subprocess is still running."""
        if self._process is None:
            return False
        return self._process.poll() is None

    # 返回启动时保存的 master 地址与元数据端口，供其他 actor 构造连接配置。
    def get_info(self) -> dict:
        """Return the master address and metadata port."""
        return self._info

    # 给 stdout 或 stderr 启一个守护读取线程，持续消费管道以避免子进程因日志塞满而阻塞。
    def _start_log_thread(self, stream, name: str) -> None:
        """Start a daemon thread that reads lines from stream and logs them."""

        # 逐行解码子进程输出，替换无效 UTF-8 并去尾换行，带通道名写入调试日志。
        def _reader():
            for line in stream:
                if isinstance(line, bytes):
                    line = line.decode("utf-8", errors="replace")
                line = line.rstrip("\n")
                if line:
                    logger.debug(f"[mooncake_master {name}] {line}")

        t = threading.Thread(target=_reader, daemon=True)
        t.start()

    # 向整个进程组发送 SIGTERM 并限时等待，失败再 SIGKILL，防止只杀包装脚本留下真实服务。
    def shutdown(self):
        """Gracefully terminate the subprocess and its entire process group."""
        if self._process is not None and self._process.poll() is None:
            try:
                os.killpg(self._process.pid, signal.SIGTERM)
                self._process.wait(timeout=5)
            except Exception:
                try:
                    os.killpg(self._process.pid, signal.SIGKILL)
                except Exception:
                    pass
            self._process = None

    # 析构兜底终止仍存活的进程组，避免 actor 回收后 master 成为孤儿进程。
    def __del__(self):
        process = getattr(self, "_process", None)
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=5)
            except Exception:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except Exception:
                    pass


# 分别解析 master 和元数据地址并进行 TCP connect，失败在加载大模型前给出具体端点错误。
def check_mooncake_master_available(
    master_server_address: str,
    metadata_server: str,
    timeout: float = 5.0,
) -> None:
    """Verify mooncake master services are reachable.

    Probes the gRPC endpoint via TCP connect and the HTTP metadata endpoint
    so actors fail fast with a clear error before expensive model loading.

    Args:
        master_server_address: gRPC address, e.g. "10.1.2.3:50051".
        metadata_server: HTTP metadata URL, e.g. "http://10.1.2.3:8090/metadata".
        timeout: Per-probe connection timeout in seconds.

    Raises:
        RuntimeError: If either service is unreachable.
    """
    # gRPC port check (TCP connect)
    try:
        grpc_host, grpc_port_str = master_server_address.rsplit(":", 1)
        grpc_port = int(grpc_port_str)
    except ValueError as exc:
        raise RuntimeError(f"Invalid master_server_address {master_server_address!r}") from exc

    try:
        with socket.create_connection((grpc_host, grpc_port), timeout=timeout):
            pass
    except OSError as exc:
        raise RuntimeError(
            f"Mooncake master gRPC unreachable at {master_server_address}: {exc}"
        ) from exc

    # HTTP metadata server check (TCP connect to parsed host:port)
    parsed = urlparse(metadata_server)
    http_host = parsed.hostname
    http_port = parsed.port or (443 if parsed.scheme == "https" else 80)
    if http_host is None:
        raise RuntimeError(f"Cannot parse host from metadata_server URL: {metadata_server!r}")

    try:
        with socket.create_connection((http_host, http_port), timeout=timeout):
            pass
    except OSError as exc:
        raise RuntimeError(
            f"Mooncake metadata server unreachable at {metadata_server}: {exc}"
        ) from exc

    logger.info(
        "Mooncake services reachable: master=%s, metadata=%s",
        master_server_address,
        metadata_server,
    )


# 解析或自动选择地址，按 host 把 actor 固定到目标节点，启动成功后把实际端口回写 args 并注册退出清理。
def launch_mooncake_master(args):
    """Launch the mooncake master as a Ray actor.

    Auto-resolves master_server_address and metadata_port if not configured.
    When master_server_address specifies a host IP, pins the actor to that node so the
    mooncake master process starts on the intended machine.
    Writes resolved values back to args for downstream code.

    Args:
        args: Arguments namespace with mooncake_master_server_address, mooncake_metadata_port, etc.

    Returns:
        The MooncakeMasterActor handle, or None if binary not found.
    """
    from torchspec.ray.ray_actor import node_affinity_for_ip

    master_addr = getattr(args, "mooncake_master_server_address", None)
    scheduling_strategy = None

    if master_addr is None:
        host = RayActor.get_node_ip()
        port = RayActor.find_free_port(start_port=random.randint(51000, 52000))
        master_addr = f"{host}:{port}"
        args.mooncake_master_server_address = master_addr
        logger.info(f"Auto-resolved mooncake master_server_address: {master_addr}")
    else:
        if ":" in master_addr:
            host = master_addr.split(":")[0]
            port = int(master_addr.split(":")[1])
        else:
            host = master_addr
            port = getattr(args, "mooncake_master_port", 50051)

    # Pin actor to the user-specified node so the master starts on the right machine.
    # 地址包含指定节点时，把服务 actor 放到那台机器；否则发布的地址与真正进程位置可能不一致。
    scheduling_strategy = node_affinity_for_ip(host, name="mooncake_master")

    http_port = getattr(args, "mooncake_metadata_port", None) or getattr(
        args, "mooncake_http_port", None
    )
    if http_port is None:
        http_port = RayActor.find_free_port(start_port=random.randint(8100, 9100))
        args.mooncake_metadata_port = http_port
        logger.info(f"Auto-resolved mooncake metadata_port: {http_port}")
    http_host = getattr(args, "mooncake_http_host", "0.0.0.0")

    # Check binary existence before creating the actor
    mooncake_bin = resolve_mooncake_master_bin()
    if not os.path.exists(mooncake_bin):
        logger.warning(f"Binary not found at {mooncake_bin}, skipping launch")
        return None

    # master 不消耗 GPU；把 TorchSpec 所需环境传播到远端，二进制和依赖必须在目标节点可用。
    RemoteActor = ray.remote(num_cpus=0, runtime_env={"env_vars": get_torchspec_env_vars()})(
        MooncakeMaster
    )
    actor_options = {"name": "mooncake_master"}
    if scheduling_strategy is not None:
        actor_options["scheduling_strategy"] = scheduling_strategy
    actor = RemoteActor.options(**actor_options).remote()

    kv_lease_ttl_s = getattr(args, "mooncake_kv_lease_ttl_s", 5.0)

    try:
        info = ray.get(actor.start.remote(port, http_port, http_host, kv_lease_ttl_s))
        # Write back resolved values (actor may have updated host from node IP)
        args.mooncake_master_server_address = info["master_addr"]
        args.mooncake_metadata_port = info["metadata_port"]
        logger.info(f"mooncake master actor started: {info}")
    except Exception as e:
        logger.error(f"Failed to launch mooncake master actor: {e}")
        return None

    # 保留 actor 引用用于生命周期管理，再注册进程退出钩子，防止局部变量消失导致服务过早回收。
    args._mooncake_master_actor = actor

    # 进程退出时最多等待十秒远程 shutdown；清理异常被忽略，避免退出钩子遮盖原始失败。
    def _cleanup():
        try:
            ray.get(actor.shutdown.remote(), timeout=10)
        except Exception:
            pass

    atexit.register(_cleanup)

    return actor
