#!/usr/bin/env python3
# 检查多节点训练的网卡、RDMA 设备和连通性，并可实际运行每 HCA 的本地带宽测试。
# 读取依赖 Linux sysfs/ip/可选 ibv_devinfo；存在 RAY_ADDRESS 时会连接集群并创建探测 actor 和 TCP 监听端口。

"""Report network topology and status for TorchSpec multi-node setups.

Checks:
  1. Available RDMA devices and their link rate/state/layer on each node.
  2. Network interfaces suitable for NCCL_SOCKET_IFNAME on each node.
  3. RDMA data-plane bandwidth per HCA via ib_write_bw loopback (requires perftest).
  4. Pairwise TCP connectivity between all nodes in the Ray cluster.

Usage (local node only):
    python tools/check_network_topology.py

Usage (full Ray cluster):
    RAY_ADDRESS=<head-node-ip>:<port> python tools/check_network_topology.py

Environment variables:
    RAY_ADDRESS               Ray cluster address (e.g. 10.0.0.1:6379).
                              If unset, only the local node is checked.
    TORCHSPEC_PROBE_TIMEOUT   TCP connect timeout in seconds for the
                              pairwise connectivity test. Default: 5.0.
                              Increase this for high-latency networks.

Dependencies for bandwidth test:
    ib_write_bw from the perftest package (apt install perftest).
    If not installed, the bandwidth section is skipped gracefully.
"""

import os
import socket
import subprocess
import time


# 读取一个设备属性文件并去空白；不可读时返回空串，让报告仍能展示其他可用字段。
def _read_sysfs(path: str) -> str:
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return ""


# 优先遍历 Linux RDMA sysfs 的每端口状态，没有目录时尝试解析 ibv_devinfo，缺工具或失败则返回空列表。
def get_rdma_devices() -> list[dict]:
    ib_root = "/sys/class/infiniband"
    devices = []

    # Linux sysfs 不存在时尝试用户态工具；这不保证非 Linux 平台具备等价能力，可能正常返回空结果。
    if not os.path.isdir(ib_root):
        try:
            result = subprocess.run(["ibv_devinfo"], capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            return devices
        if result.returncode != 0:
            return devices
        current_hca: str | None = None
        current: dict | None = None
        for line in result.stdout.splitlines():
            line = line.strip()
            if line.startswith("hca_id:"):
                if current:
                    devices.append(current)
                    current = None
                current_hca = line.split()[-1]
            elif current_hca and line.startswith("port:"):
                if current:
                    devices.append(current)
                current = {
                    "name": current_hca,
                    "port": int(line.split()[-1]),
                    "state": "unknown",
                    "rate": "unknown",
                    "link_layer": "unknown",
                    "phys_state": "unknown",
                }
            elif current and line.startswith("transport:"):
                current["link_layer"] = line.split()[-1]
            elif current and line.startswith("state:"):
                current["state"] = line.split(":", 1)[-1].strip()
            elif current and line.startswith("phys_state:"):
                current["phys_state"] = line.split(":", 1)[-1].strip()
            elif current and line.startswith("active_speed:"):
                current["rate"] = line.split(":", 1)[-1].strip()
        if current:
            devices.append(current)
        return devices

    # 每 HCA 可能多端口，报告单位是设备端口而非只列设备名；链路状态与速率需一起查看。
    for dev_name in sorted(os.listdir(ib_root)):
        ports_path = os.path.join(ib_root, dev_name, "ports")
        if not os.path.isdir(ports_path):
            continue
        for port_num in sorted(os.listdir(ports_path)):
            port_path = os.path.join(ports_path, port_num)
            state = _read_sysfs(os.path.join(port_path, "state"))
            rate = _read_sysfs(os.path.join(port_path, "rate"))
            link_layer = _read_sysfs(os.path.join(port_path, "link_layer"))
            phys_state = _read_sysfs(os.path.join(port_path, "phys_state"))
            devices.append(
                {
                    "name": dev_name,
                    "port": int(port_num),
                    "state": state,
                    "rate": rate,
                    "link_layer": link_layer,
                    "phys_state": phys_state,
                }
            )

    return devices


# 从每个 InfiniBand 设备关联的 net 目录收集网卡名，用于标记哪些 socket 接口属于 RDMA 设备。
def _rdma_backed_interfaces() -> set[str]:
    ib_root = "/sys/class/infiniband"
    ifaces: set[str] = set()
    if not os.path.isdir(ib_root):
        return ifaces
    for dev_name in os.listdir(ib_root):
        net_path = os.path.join(ib_root, dev_name, "device", "net")
        if os.path.isdir(net_path):
            for iface in os.listdir(net_path):
                ifaces.add(iface)
    return ifaces


# 调用 ip -4 addr 读取接口第一个 IPv4 地址，不解析全部地址或自动选择最佳路由。
def _iface_ipv4(iface: str) -> str:
    try:
        result = subprocess.run(
            ["ip", "-4", "addr", "show", iface],
            capture_output=True,
            text=True,
            timeout=5,
        )
        for line in result.stdout.splitlines():
            line = line.strip()
            if line.startswith("inet "):
                return line.split()[1].split("/")[0]
    except (OSError, subprocess.TimeoutExpired):
        pass
    return ""


# 过滤回环/容器前缀和非 up 网卡，附带 IPv4 与 RDMA 关联标志；这是候选枚举，不是 NCCL 实测。
def get_nccl_interfaces() -> list[dict]:
    net_root = "/sys/class/net"
    if not os.path.isdir(net_root):
        return []

    rdma_ifaces = _rdma_backed_interfaces()
    skip_prefixes = ("lo", "docker", "veth", "virbr", "br-")
    results = []

    for iface in sorted(os.listdir(net_root)):
        if any(iface.startswith(p) for p in skip_prefixes):
            continue
        operstate = _read_sysfs(os.path.join(net_root, iface, "operstate"))
        if operstate != "up":
            continue
        ip_addr = _iface_ipv4(iface)
        results.append(
            {
                "name": iface,
                "operstate": operstate,
                "ip": ip_addr,
                "rdma_backed": iface in rdma_ifaces,
            }
        )

    return results


# 按设备和端口打印状态、速率及链路层，缺设备时报告空结果而不启动额外测试。
def print_rdma_report(devices: list[dict], node_label: str = "local") -> None:
    print(f"\nRDMA Devices on {node_label}:")
    if not devices:
        print("  No RDMA devices found.")
        return
    col = "{:<16} {:>5}  {:<22} {:<28} {:<16} {:<20}"
    print("  " + col.format("Device", "Port", "State", "Rate", "Link Layer", "Phys State"))
    print("  " + "-" * 110)
    for d in devices:
        print(
            "  "
            + col.format(
                d["name"], d["port"], d["state"], d["rate"], d["link_layer"], d["phys_state"]
            )
        )


# 打印接口并优先推荐第一个 RDMA 关联名称；不会自动写入环境，也不保证该接口跨所有节点可达。
def print_nccl_report(ifaces: list[dict], node_label: str = "local") -> None:
    print(f"\nNetwork Interfaces for NCCL_SOCKET_IFNAME on {node_label}:")
    if not ifaces:
        print("  No suitable UP interfaces found.")
        return
    col = "{:<20} {:<16} {:<10}"
    print("  " + col.format("Interface", "IP", "RDMA-backed"))
    print("  " + "-" * 50)
    for i in ifaces:
        print("  " + col.format(i["name"], i["ip"] or "n/a", "yes" if i["rdma_backed"] else "no"))

    rdma_names = [i["name"] for i in ifaces if i["rdma_backed"]]
    candidates = rdma_names if rdma_names else [i["name"] for i in ifaces]
    if candidates:
        print(f"\n  Recommended: export NCCL_SOCKET_IFNAME={candidates[0]}")


# 逐端口尝试 TCP bind，成功立即关闭后返回；只是找到当前空闲候选，不是持续保留端口。
def _find_free_port(start: int = 18500) -> int:
    port = start
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("", port))
                return port
            except OSError:
                port += 1


# 从 HCA 关联网卡中返回首个可读 IPv4，为同节点 ib_write_bw 客户端提供地址。
def _rdma_device_ip(dev_name: str) -> str | None:
    net_path = f"/sys/class/infiniband/{dev_name}/device/net"
    if os.path.isdir(net_path):
        for iface in os.listdir(net_path):
            ip = _iface_ipv4(iface)
            if ip:
                return ip
    return None


# 从工具输出的数字结果行取第四列为平均带宽；依赖 ib_write_bw 当前文本布局。
def _parse_ib_write_bw(stdout: str) -> float | None:
    for line in stdout.splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[0].isdigit():
            try:
                return float(parts[3])
            except ValueError:
                pass
    return None


# 每 HCA 选一个 ACTIVE 端口，启动本地 server/client 做三秒写带宽测量，缺 perftest 时跳过并记录原因。
def measure_rdma_bandwidth(devices: list[dict]) -> list[dict]:
    by_device: dict[str, list[dict]] = {}
    for dev in devices:
        by_device.setdefault(dev["name"], []).append(dev)

    results = []
    tool_missing = False

    for dev_name, ports in by_device.items():
        if tool_missing:
            results.append(
                {"device": dev_name, "bw_gbps": None, "note": "ib_write_bw not installed"}
            )
            continue

        # 只选每设备的第一个 ACTIVE 端口，结果并未覆盖每个端口或聚合所有 HCA。
        active_port = next((p for p in ports if "ACTIVE" in p.get("state", "").upper()), None)
        if active_port is None:
            results.append({"device": dev_name, "bw_gbps": None, "note": "no active port found"})
            continue

        port = _find_free_port(18500)
        target_ip = _rdma_device_ip(dev_name)
        if target_ip is None:
            results.append(
                {"device": dev_name, "bw_gbps": None, "note": "could not resolve device IP"}
            )
            continue
        server = None
        try:
            # server/client 位于同一节点、同一设备；这验证本地数据路径，不能证明跨机器交换网络带宽正常。
            server = subprocess.Popen(
                [
                    "ib_write_bw",
                    "-d",
                    dev_name,
                    "-i",
                    str(active_port["port"]),
                    "--port",
                    str(port),
                    "-D",
                    "3",
                    "--report_gbits",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            time.sleep(0.5)
            # 三秒测量并设置二十秒进程超时，避免异常设备让整个诊断永久挂起。
            client_result = subprocess.run(
                [
                    "ib_write_bw",
                    "-d",
                    dev_name,
                    "-i",
                    str(active_port["port"]),
                    "--port",
                    str(port),
                    target_ip,
                    "-D",
                    "3",
                    "--report_gbits",
                ],
                capture_output=True,
                text=True,
                timeout=20,
            )
            bw = _parse_ib_write_bw(client_result.stdout)
            results.append(
                {
                    "device": dev_name,
                    "bw_gbps": bw,
                    "note": "" if bw is not None else "parse failed",
                }
            )
        except FileNotFoundError:
            tool_missing = True
            print("  ib_write_bw not found. Install it with: apt install perftest")
            results.append(
                {"device": dev_name, "bw_gbps": None, "note": "ib_write_bw not installed"}
            )
        except (OSError, subprocess.TimeoutExpired) as e:
            results.append({"device": dev_name, "bw_gbps": None, "note": str(e)})
        finally:
            # 无论正常、缺工具或超时，已启动 server 都要回收，避免占住测试端口影响下一轮。
            if server is not None:
                server.terminate()
                try:
                    server.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    server.kill()

    return results


# 打印 Gb/s 带宽和失败说明；这里测的是设备本地回环，不能当成跨节点 RDMA 吞吐。
def print_rdma_bw_report(bw_results: list[dict], node_label: str = "local") -> None:
    print(f"\nRDMA Bandwidth (ib_write_bw loopback) on {node_label}:")
    if not bw_results:
        print("  No active RDMA devices to test.")
        return
    col = "{:<16} {:<18} {}"
    print("  " + col.format("Device", "BW average", "Note"))
    print("  " + "-" * 55)
    for r in bw_results:
        bw_str = f"{r['bw_gbps']:.2f} Gb/s" if r["bw_gbps"] is not None else "n/a"
        print("  " + col.format(r["device"], bw_str, r.get("note", "")))


# 组合设备枚举、NCCL 候选和真实带宽测试；名字虽叫 info，执行仍可能启动 perftest 子进程。
def _local_probe_info() -> dict:
    rdma = get_rdma_devices()
    return {
        "rdma_devices": rdma,
        "nccl_interfaces": get_nccl_interfaces(),
        "rdma_bandwidth": measure_rdma_bandwidth(rdma),
    }


# 从 TORCHSPEC_PROBE_TIMEOUT 读取秒数，供 TCP 建连和服务接受窗口使用。
def _connect_timeout() -> float:
    return float(os.environ.get("TORCHSPEC_PROBE_TIMEOUT", "5.0"))


# 在所有 IPv4 地址监听系统分配的临时 TCP 端口，返回 socket 与端口交给远端探测调度。
def _tcp_server_listen() -> tuple[socket.socket, int]:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("0.0.0.0", 0))
    port = s.getsockname()[1]
    s.listen(64)
    s.settimeout(_connect_timeout() * 2)
    return s, port


# 测一次 TCP connect 的墙钟时间并立即关闭；返回的毫秒不是应用层往返消息或吞吐测试。
def _tcp_probe(target_ip: str, port: int) -> tuple[bool, float]:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(_connect_timeout())
    t0 = time.monotonic()
    try:
        s.connect((target_ip, port))
        rtt = (time.monotonic() - t0) * 1000
        return True, rtt
    except OSError:
        return False, -1.0
    finally:
        s.close()


# 枚举本机接口并实际尝试 RDMA 带宽，顺序输出三种报告；不连接 Ray。
def run_local() -> None:
    rdma = get_rdma_devices()
    ifaces = get_nccl_interfaces()
    bw = measure_rdma_bandwidth(rdma)
    print_rdma_report(rdma)
    print_nccl_report(ifaces)
    print_rdma_bw_report(bw)


# 连接现有 Ray 集群，在每活跃节点固定一个 actor，收集本机信息后建立有向两两 TCP 连通矩阵。
def run_cluster() -> None:
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    ray_address = os.environ.get("RAY_ADDRESS", "auto")
    ray.init(address=ray_address, ignore_reinit_error=True)

    live_nodes = [n for n in ray.nodes() if n.get("Alive", False)]
    if not live_nodes:
        print("No live Ray nodes found.")
        return

    print(f"Ray cluster: {len(live_nodes)} live node(s)")

    # actor 需要同时 accept 和 probe；若只有串行方法执行，accept 会阻塞它主动连接其他节点。
    max_concurrency = len(live_nodes) + 2

    # 节点固定的并发探测对象，允许接受连接与发起探测同时进行，避免串行 actor 被 accept 阻塞。
    @ray.remote(num_cpus=0, max_concurrency=max_concurrency)
    class NetworkProbeActor:
        # 在 actor 所在节点执行完整本机探测，包括可选 perftest 子进程。
        def gather_info(self) -> dict:
            return _local_probe_info()

        # 创建并保存该节点的监听 socket，端口回传驱动进程用于构造其他节点的目标地址。
        def open_server(self) -> int:
            self._server, port = _tcp_server_listen()
            return port

        # 在总期限内接受预期数量的连接并关闭它们，最后关闭监听器；未收齐不无限等待。
        def accept_all(self, count: int) -> None:
            deadline = time.monotonic() + _connect_timeout() * 2
            accepted = 0
            while accepted < count:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._server.settimeout(remaining)
                try:
                    conn, _ = self._server.accept()
                    conn.close()
                    accepted += 1
                except OSError:
                    break
            self._server.close()

        # 从当前节点向指定目标发起 TCP 建连，保持连通矩阵的方向性。
        def probe(self, target_ip: str, port: int) -> tuple[bool, float]:
            return _tcp_probe(target_ip, port)

    actors = []
    for node in live_nodes:
        node_id = node["NodeID"]
        # 硬节点亲和保证每个 live node 真正执行一次探测，而不是 Ray 把多个 actor 全排到同一台机器。
        strategy = NodeAffinitySchedulingStrategy(node_id=node_id, soft=False)
        actor = NetworkProbeActor.options(scheduling_strategy=strategy).remote()
        actors.append((node["NodeManagerAddress"], actor))

    infos = ray.get([a.gather_info.remote() for _, a in actors])

    for (node_ip, _), info in zip(actors, infos):
        print_rdma_report(info["rdma_devices"], node_label=node_ip)
        print_nccl_report(info["nccl_interfaces"], node_label=node_ip)
        print_rdma_bw_report(info["rdma_bandwidth"], node_label=node_ip)

    print("\nPairwise TCP Connectivity Matrix:")
    n = len(actors)
    node_ips_list = [ip for ip, _ in actors]

    # 先让所有监听器建立，再发 accept 和连接任务，降低探测早于监听而出现假失败的概率。
    server_ports = ray.get([a.open_server.remote() for _, a in actors])

    accept_futures = [a.accept_all.remote(n - 1) for _, a in actors]

    # 测有向边 i→j，两节点相反方向可能受不同防火墙/路由限制，不能只测一半矩阵。
    probe_futures = {}
    for i, (_, src_actor) in enumerate(actors):
        for j, (tgt_ip, tgt_port) in enumerate(zip(node_ips_list, server_ports)):
            if i == j:
                continue
            fut = src_actor.probe.remote(tgt_ip, tgt_port)
            probe_futures[(i, j)] = fut

    keys = list(probe_futures.keys())
    # 一次汇合所有连接结果；RTT 标签实际对应 connect 建连时间，并无应用层 echo 往返。
    probe_results = dict(zip(keys, ray.get([probe_futures[k] for k in keys])))
    ray.get(accept_futures)

    header = "{:<18}".format("src \\ dst")
    for ip in node_ips_list:
        header += "{:>22}".format(ip)
    print(header)
    print("-" * (18 + 22 * n))
    for i in range(n):
        row = "{:<18}".format(node_ips_list[i])
        for j in range(n):
            if i == j:
                row += "{:>22}".format("self")
            else:
                ok, rtt = probe_results[(i, j)]
                cell = f"OK {rtt:.1f}ms" if ok else "FAIL"
                row += "{:>22}".format(cell)
        print(row)


if __name__ == "__main__":
    ray_address = os.environ.get("RAY_ADDRESS", "")
    if ray_address:
        run_cluster()
    else:
        run_local()
