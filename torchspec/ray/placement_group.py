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

# 中文导读：把“需要多少训练/推理 GPU”映射为 Ray 集群上的具体资源位置。
# placement group 是一起预留的一组 bundle（CPU/GPU 资源配额），并不是 PyTorch 的通信 group。
# 先探测实际节点和 GPU，再稳定排序并分给角色；这样重启时更容易复用本地权重和编译缓存。
# 逻辑 rank、bundle 索引、物理 GPU ID、进程可见 CUDA 序号是不同坐标，阅读时要分别追踪。

from __future__ import annotations

import math
import os
import socket
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from torchspec.ray.train_group import RayTrainGroup
from torchspec.utils.logging import logger

# Ray exposes a tiny "node:<ip>" resource on each node. Requiring a fractional
# amount pins a bundle to that node without consuming a full logical resource.
_NODE_RESOURCE_EPSILON = 0.001
_CUSTOM_PLACEMENT_FIELDS = (
    "training_node_ips",
    "inference_node_ips",
    "training_node_selectors",
    "inference_node_selectors",
)


# 不可变节点选择条件：IP 或标签选择器；统一表示后便于验证训练与推理是否要求同一组机器。
@dataclass(frozen=True)
class _NodeConstraint:
    ip: str | None = None
    label_selector: tuple[tuple[str, str], ...] = ()

    # 为日志返回 IP 或标签字典文本，不改变实际调度约束。
    @property
    def selector_for_log(self) -> str:
        if self.ip is not None:
            return self.ip
        return str(dict(self.label_selector))

    # IP 约束转换为很小的 node:<ip> 资源需求，借此固定节点而不占满该逻辑资源。
    def to_bundle_resource(self) -> dict[str, float]:
        if self.ip is None:
            return {}
        return {_node_ip_resource(self.ip): _NODE_RESOURCE_EPSILON}

    # 把稳定存储的键值元组恢复为 Ray 接收的标签字典。
    def to_label_selector(self) -> dict[str, str]:
        return dict(self.label_selector)


# 临时 GPU 探测 actor：在每个已预留 bundle 上运行，得到 Ray 实际分配的 GPU。
@ray.remote(num_gpus=1)
class InfoActor:
    # 返回所在节点 IP 和 Ray 分配的第一个 GPU ID，供重排逻辑 rank。
    def get_ip_and_gpu_id(self):
        return ray.util.get_node_ip_address(), ray.get_gpu_ids()[0]


# 优先按数值 IP 再按 GPU ID 排序；主机名先解析，不能解析时用字符码形成稳定顺序。
# 数值排序避免字符串把 10.0.0.10 放在 10.0.0.2 前面。
def sort_key(x):
    _index, node_identifier, gpu_id = x
    # Sort by node IP number and then by GPU ID
    try:
        # try to parse it as an IP address.
        ip_address = node_identifier
        node_ip_parts = list(map(int, ip_address.split(".")))
    except ValueError:
        # Try to resolve the hostname to an IP address.
        try:
            ip_address = socket.gethostbyname(node_identifier)
            node_ip_parts = list(map(int, ip_address.split(".")))
        except (socket.gaierror, TypeError):
            # Instead, we convert each character of the original identifier string
            # to its ASCII value. This provides a stable and consistent numerical
            # representation that allows for sorting.
            node_ip_parts = [ord(c) for c in node_identifier]

    return (node_ip_parts, gpu_id)


# 把 None、空列表和空字典视为未配置，用于判断自定义节点字段是否真的启用。
def _has_value(value) -> bool:
    return value is not None and value != [] and value != {}


# 检查任一训练/推理节点 IP 或标签列表是否非空。
def _has_custom_placement_fields(args) -> bool:
    return any(_has_value(getattr(args, field, None)) for field in _CUSTOM_PLACEMENT_FIELDS)


# 要求节点约束必须配合 custom 策略，防止用户以为指定了机器但实际仍按默认 PACK 调度。
def _validate_custom_strategy_usage(args) -> None:
    placement_strategy = getattr(args, "placement_strategy", "training_first")
    if placement_strategy != "custom" and _has_custom_placement_fields(args):
        raise ValueError(
            "Custom placement fields require training.placement_strategy=custom. "
            f"Got placement_strategy={placement_strategy!r}."
        )


# 校验配置为序列并转为普通列表；特别拒绝字符串，避免把一个 IP 当成逐字符节点列表。
def _as_list(value, field_name: str) -> list:
    if value is None:
        return []
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError(f"{field_name} must be a list, got {type(value).__name__}")
    return list(value)


# 每个角色只能选 IP 列表或标签列表之一，校验非空内容后规范化为 _NodeConstraint。
# 标签键值排序让语义相同但书写顺序不同的选择器能做稳定比较。
def _normalize_node_constraints(
    args, role: str, *, required: bool = False
) -> list[_NodeConstraint]:
    ip_field = f"{role}_node_ips"
    selector_field = f"{role}_node_selectors"
    node_ips = _as_list(getattr(args, ip_field, None), ip_field)
    node_selectors = _as_list(getattr(args, selector_field, None), selector_field)

    if node_ips and node_selectors:
        raise ValueError(f"Set only one of {ip_field} or {selector_field}, not both")
    if not node_ips and not node_selectors:
        if required:
            raise ValueError(
                f"training.placement_strategy=custom requires {ip_field} or {selector_field}"
            )
        return []

    if node_ips:
        constraints = []
        for ip in node_ips:
            if not isinstance(ip, str) or not ip:
                raise ValueError(f"{ip_field} entries must be non-empty strings")
            constraints.append(_NodeConstraint(ip=ip))
        return constraints

    constraints = []
    for selector in node_selectors:
        if not isinstance(selector, Mapping) or not selector:
            raise ValueError(f"{selector_field} entries must be non-empty mappings")
        normalized = tuple(sorted((str(key), str(value)) for key, value in selector.items()))
        constraints.append(_NodeConstraint(label_selector=normalized))
    return constraints


# 生成 Ray 的节点资源名；它是调度标签，不是连接端口或网络地址 URI。
def _node_ip_resource(ip: str) -> str:
    return f"node:{ip}"


# 向上取整计算所需节点数，允许最后一节点只用部分 GPU；负数和无效每节点卡数提前报错。
def _expected_node_count(total_gpus: int, gpus_per_node: int, role: str) -> int:
    if total_gpus < 0:
        raise ValueError(f"{role} total GPUs must be non-negative, got {total_gpus}")
    if total_gpus == 0:
        return 0
    if gpus_per_node <= 0:
        raise ValueError(f"{role} GPUs per node must be positive, got {gpus_per_node}")
    return math.ceil(total_gpus / gpus_per_node)


# 按用户节点顺序给每张 GPU 建一个 bundle，最后一节点按剩余卡数分配。
# 同时保存节点序号，实际探测后仍按用户顺序恢复逻辑 rank，不改成 IP 自然排序。
def _build_custom_bundles(
    role: str,
    constraints: list[_NodeConstraint],
    total_gpus: int,
    gpus_per_node: int,
) -> tuple[list[dict[str, float]], list[dict[str, str]], list[int]]:
    """Build Ray bundles for nodes in user-provided order.

    ``node_group_indices`` records the configured node ordinal for each bundle.
    After Ray schedules the placement group, InfoActor probes actual nodes and
    GPU ids; the ordinal lets us restore user node order while still sorting
    GPUs within a node by physical GPU id.
    """
    expected_nodes = _expected_node_count(total_gpus, gpus_per_node, role)
    if len(constraints) != expected_nodes:
        raise ValueError(
            f"{role} custom placement expected {expected_nodes} node(s) for "
            f"{total_gpus} GPU(s) with {gpus_per_node} GPU(s) per node, "
            f"got {len(constraints)}"
        )

    bundles: list[dict[str, float]] = []
    bundle_label_selectors: list[dict[str, str]] = []
    node_group_indices: list[int] = []
    remaining = total_gpus

    for node_index, constraint in enumerate(constraints):
        gpus_on_node = min(gpus_per_node, remaining)
        remaining -= gpus_on_node

        for _ in range(gpus_on_node):
            bundle: dict[str, float] = {"GPU": 1, "CPU": 1, **constraint.to_bundle_resource()}
            bundles.append(bundle)
            bundle_label_selectors.append(constraint.to_label_selector())
            node_group_indices.append(node_index)

    return bundles, bundle_label_selectors, node_group_indices


# 全部没有标签时返回 None，让旧版 Ray 走不带新参数的调用；否则保留每 bundle 对应关系。
def _merge_bundle_label_selectors(
    selectors: list[dict[str, str]],
) -> list[dict[str, str]] | None:
    return selectors if any(selector for selector in selectors) else None


# 封装 Ray 创建调用；安装版本不支持标签参数时解释兼容问题，不静默丢弃用户的节点约束。
def _placement_group(
    bundles: list[dict[str, float]],
    *,
    strategy: str,
    name: str | None,
    bundle_label_selector: list[dict[str, str]] | None = None,
):
    kwargs = {"bundles": bundles, "strategy": strategy, "name": name}
    if bundle_label_selector is not None:
        kwargs["bundle_label_selector"] = bundle_label_selector

    try:
        return placement_group(**kwargs)
    except TypeError as e:
        if bundle_label_selector is not None and "bundle_label_selector" in str(e):
            raise RuntimeError(
                "Ray bundle_label_selector is not supported by the installed Ray version. "
                "Use training_node_ips/inference_node_ips or upgrade Ray."
            ) from e
        raise


# 默认按节点/GPU 稳定排序；custom 则先按用户配置节点序号，再按节点内 GPU ID。
def _sort_probed_bundle_infos(gpu_ids, node_group_indices: list[int] | None = None):
    """Sort probed bundles by default topology or explicit user node order."""
    bundle_infos = [(i, gpu_ids[i][0], gpu_ids[i][1]) for i in range(len(gpu_ids))]
    if node_group_indices is None:
        return sorted(bundle_infos, key=sort_key)
    if len(node_group_indices) != len(gpu_ids):
        raise ValueError(
            f"node_group_indices length ({len(node_group_indices)}) must match "
            f"bundle count ({len(gpu_ids)})"
        )
    return sorted(bundle_infos, key=lambda info: (node_group_indices[info[0]], info[2]))


# 预留 bundle 并等待 ready，再用临时 actor 探测实际位置，释放探测 actor 后返回重排映射。
# 返回 (pg, 逻辑索引到原 bundle 的列表, 对应物理 GPU ID)，调用者不能直接把 rank 当原 bundle 索引。
def _create_placement_group(
    num_gpus,
    strategy="PACK",
    name=None,
    *,
    bundles: list[dict[str, float]] | None = None,
    bundle_label_selector: list[dict[str, str]] | None = None,
    node_group_indices: list[int] | None = None,
):
    """Create a placement group with the specified number of GPUs."""
    if bundles is None:
        bundles = [{"GPU": 1, "CPU": 1} for _ in range(num_gpus)]
    elif len(bundles) != num_gpus:
        raise ValueError(f"num_gpus={num_gpus} does not match bundle count={len(bundles)}")

    pg = _placement_group(
        bundles,
        strategy=strategy,
        name=name,
        bundle_label_selector=bundle_label_selector,
    )
    num_bundles = len(bundles)

    ray.get(pg.ready())
    # use info actor to get the GPU id
    info_actors = []
    for i in range(num_bundles):
        info_actors.append(
            InfoActor.options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=i,
                )
            ).remote()
        )
    gpu_ids = ray.get([actor.get_ip_and_gpu_id.remote() for actor in info_actors])
    for actor in info_actors:
        ray.kill(actor)

    sorted_bundle_infos = _sort_probed_bundle_infos(gpu_ids, node_group_indices)
    pg_reordered_bundle_indices = [info[0] for info in sorted_bundle_infos]
    # Map from logical index -> physical GPU ID
    pg_reordered_gpu_ids = [gpu_ids[info[0]][1] for info in sorted_bundle_infos]

    for i in range(num_bundles):
        actual_bundle_index = pg_reordered_bundle_indices[i]
        logger.info(
            f"  bundle {i:4}, actual_bundle_index: {actual_bundle_index:4}, "
            f"node: {gpu_ids[actual_bundle_index][0]}, gpu: {gpu_ids[actual_bundle_index][1]}"
        )

    return pg, pg_reordered_bundle_indices, pg_reordered_gpu_ids


# 默认尝试连接已有集群，auto 找不到才起本地实例；显式 RAY_ADDRESS 连接失败则报错。
# 这样避免集群地址写错时悄悄在本机开训。
def _ensure_ray_initialized():
    """Connect to an existing Ray cluster, or start a local instance as fallback."""
    if ray.is_initialized():
        return

    ray_address = os.environ.get("RAY_ADDRESS", "auto")
    try:
        ray.init(address=ray_address, ignore_reinit_error=True)
        logger.info(f"Connected to Ray cluster at {ray_address}")
    except ConnectionError:
        if ray_address == "auto":
            logger.warning("No existing Ray cluster found, starting a local instance")
            ray.init(ignore_reinit_error=True)
            return
        raise RuntimeError(
            f"Failed to connect to Ray cluster at {ray_address}. "
            "Refusing to fall back to a local Ray instance when RAY_ADDRESS is explicitly set."
        ) from None


# 隔离部署按训练+推理卡数求和；colocate 共卡按两者最大值估算所需集群规模。
def _get_expected_gpu_count(args) -> int:
    training_gpus = args.training_num_nodes * args.training_num_gpus_per_node
    inference_gpus = getattr(args, "inference_num_gpus", 0)
    if getattr(args, "colocate", False):
        return max(training_gpus, inference_gpus)
    return training_gpus + inference_gpus


# 等待已加入 Ray 集群的 GPU 总量达到要求，超时提示检查节点注册。
# cluster_resources 是集群总资源，不等于此刻空闲资源；真正预留还要等待 pg.ready()。
def _wait_for_gpu_resources(expected_gpus: int, timeout: int = 300, poll_interval: int = 5):
    """Block until the Ray cluster has at least ``expected_gpus`` GPUs."""
    available = int(ray.cluster_resources().get("GPU", 0))
    if available >= expected_gpus:
        logger.info(f"Ray cluster has {available} GPUs (need {expected_gpus})")
        return

    logger.info(f"Waiting for {expected_gpus} GPUs (currently {available}), timeout={timeout}s...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(poll_interval)
        available = int(ray.cluster_resources().get("GPU", 0))
        logger.info(f"Ray cluster GPUs: {available}/{expected_gpus}")
        if available >= expected_gpus:
            logger.info(f"All {expected_gpus} GPUs available")
            return

    raise RuntimeError(
        f"Timed out waiting for GPUs: {available}/{expected_gpus} after {timeout}s. "
        f"Check that all Ray worker nodes have joined the cluster."
    )


# 将训练和推理的自定义 bundle 合到一个 PG，节点序号错开，排序后再拆回两个角色。
def _create_custom_unified_placement_group(args, num_training_gpus: int, num_inference_gpus: int):
    training_constraints = _normalize_node_constraints(
        args, "training", required=num_training_gpus > 0
    )
    inference_constraints = _normalize_node_constraints(
        args, "inference", required=num_inference_gpus > 0
    )

    training_bundles, training_selectors, training_groups = _build_custom_bundles(
        "training",
        training_constraints,
        num_training_gpus,
        args.training_num_gpus_per_node,
    )
    inference_bundles, inference_selectors, inference_groups = _build_custom_bundles(
        "inference",
        inference_constraints,
        num_inference_gpus,
        args.inference_num_gpus_per_node,
    )

    node_group_offset = len(training_constraints)
    node_group_indices = training_groups + [
        node_group_offset + node_group_index for node_group_index in inference_groups
    ]
    bundles = training_bundles + inference_bundles
    bundle_label_selector = _merge_bundle_label_selectors(training_selectors + inference_selectors)

    total_gpus = num_training_gpus + num_inference_gpus
    logger.info(
        "Creating custom unified placement group with "
        f"{total_gpus} GPUs ({num_training_gpus} training + {num_inference_gpus} inference); "
        f"training nodes={[constraint.selector_for_log for constraint in training_constraints]}, "
        f"inference nodes={[constraint.selector_for_log for constraint in inference_constraints]}"
    )

    pg, sorted_bundle_indices, sorted_gpu_ids = _create_placement_group(
        total_gpus,
        strategy="PACK",
        name="custom_unified_pg",
        bundles=bundles,
        bundle_label_selector=bundle_label_selector,
        node_group_indices=node_group_indices,
    )

    training_bundle_indices = sorted_bundle_indices[:num_training_gpus]
    training_gpu_ids = sorted_gpu_ids[:num_training_gpus]
    inference_bundle_indices = sorted_bundle_indices[num_training_gpus:]
    inference_gpu_ids = sorted_gpu_ids[num_training_gpus:]

    logger.info(
        f"Placement (strategy=custom): "
        f"training bundles={training_bundle_indices}, "
        f"inference bundles={inference_bundle_indices}"
    )

    return {
        "training": (pg, training_bundle_indices, training_gpu_ids),
        "inference": (pg, inference_bundle_indices, inference_gpu_ids),
    }


# 共卡时两种角色必须约束一致，或只提供一方约束，否则不能保证它们使用同一组设备。
def _get_custom_colocated_constraints(args) -> tuple[str, list[_NodeConstraint]]:
    training_constraints = _normalize_node_constraints(args, "training")
    inference_constraints = _normalize_node_constraints(args, "inference")
    if (
        training_constraints
        and inference_constraints
        and training_constraints != inference_constraints
    ):
        raise ValueError(
            "custom colocate placement requires training and inference node constraints "
            "to match, or only one role's constraints to be set"
        )

    if training_constraints:
        role = "training"
        constraints = training_constraints
    elif inference_constraints:
        role = "inference"
        constraints = inference_constraints
    else:
        raise ValueError(
            "training.placement_strategy=custom with colocate=True requires training_node_* "
            "or inference_node_* constraints"
        )

    return role, constraints


# 在真实申请资源前用同一 bundle 构造逻辑验证节点数和拓扑，尽早暴露配置错误。
def _validate_custom_placement_constraints(args) -> None:
    if getattr(args, "placement_strategy", "training_first") != "custom":
        return

    if args.colocate:
        num_gpus = args.training_num_nodes * args.training_num_gpus_per_node
        _role, constraints = _get_custom_colocated_constraints(args)
        _build_custom_bundles(
            "colocate",
            constraints,
            num_gpus,
            args.training_num_gpus_per_node,
        )
        return

    num_training_gpus = args.training_num_nodes * args.training_num_gpus_per_node
    num_inference_gpus = args.inference_num_gpus

    training_constraints = _normalize_node_constraints(
        args, "training", required=num_training_gpus > 0
    )
    _build_custom_bundles(
        "training",
        training_constraints,
        num_training_gpus,
        args.training_num_gpus_per_node,
    )

    inference_constraints = _normalize_node_constraints(
        args, "inference", required=num_inference_gpus > 0
    )
    _build_custom_bundles(
        "inference",
        inference_constraints,
        num_inference_gpus,
        args.inference_num_gpus_per_node,
    )


# 按训练拓扑建立共享 PG；节点约束可来自任一角色，但数量以实际共用卡数校验。
def _create_custom_colocated_placement_group(args, num_gpus: int):
    role, constraints = _get_custom_colocated_constraints(args)
    # Colocate creates one shared placement group using the training topology.
    # Either role's node constraints may select the nodes, but node count
    # validation must use the topology that determines ``num_gpus``.
    bundles, bundle_label_selectors, node_group_indices = _build_custom_bundles(
        "colocate",
        constraints,
        num_gpus,
        args.training_num_gpus_per_node,
    )
    logger.info(
        f"Creating custom colocated placement group with {num_gpus} GPU(s) "
        f"using {role} constraints on "
        f"{[constraint.selector_for_log for constraint in constraints]}"
    )
    return _create_placement_group(
        num_gpus,
        strategy="PACK",
        name="custom_colocate_pg",
        bundles=bundles,
        bundle_label_selector=_merge_bundle_label_selectors(bundle_label_selectors),
        node_group_indices=node_group_indices,
    )


# 总入口：支持只建训练/推理资源、共卡或分离部署；返回统一的角色→资源映射。
# 普通分离模式一次申请总资源再切分，training_first/inference_first 决定稳定顺序中的先后份额。
# 单角色模式仍返回两键字典，未使用角色对应空 bundle 列表，方便 offline/物化复用调用接口。
def create_placement_groups(args, roles: set[str] | None = None):
    """Create placement groups for the requested training/inference roles."""
    if roles is None:
        roles = {"training", "inference"}
    roles = frozenset(roles)
    unknown = roles - {"training", "inference"}
    if not roles or unknown:
        raise ValueError(f"Invalid placement roles: {sorted(roles)}")

    _ensure_ray_initialized()

    if len(roles) == 1:
        if getattr(
            args, "placement_strategy", "training_first"
        ) == "custom" or _has_custom_placement_fields(args):
            raise ValueError(
                "Custom placement is only supported when training and inference "
                "roles are created together"
            )

        role = next(iter(roles))
        if role == "training":
            num_gpus = args.training_num_nodes * args.training_num_gpus_per_node
        else:
            num_gpus = args.inference_num_gpus
        if not isinstance(num_gpus, int) or num_gpus <= 0:
            raise ValueError(f"{role.capitalize()} placement requires a positive GPU count")

        _wait_for_gpu_resources(num_gpus)
        logger.info("Creating %s-only placement with %d GPUs...", role, num_gpus)
        pg, bundle_indices, gpu_ids = _create_placement_group(
            num_gpus, strategy="PACK", name=f"{role}_pg"
        )
        empty = (pg, [], [])
        result = {"training": empty, "inference": empty}
        result[role] = (pg, bundle_indices, gpu_ids)
        return result

    _validate_custom_strategy_usage(args)
    _validate_custom_placement_constraints(args)
    _wait_for_gpu_resources(_get_expected_gpu_count(args))
    placement_strategy = getattr(args, "placement_strategy", "training_first")

    if args.colocate:
        num_gpus = args.training_num_nodes * args.training_num_gpus_per_node
        logger.info(f"Creating colocated placement group with {num_gpus} GPUs...")
        if placement_strategy == "custom":
            pg, bundle_indices, gpu_ids = _create_custom_colocated_placement_group(args, num_gpus)
        else:
            pg, bundle_indices, gpu_ids = _create_placement_group(
                num_gpus, strategy="PACK", name="colocate_pg"
            )
        return {
            "training": (pg, bundle_indices, gpu_ids),
            "inference": (pg, bundle_indices, gpu_ids),
        }

    num_training_gpus = args.training_num_nodes * args.training_num_gpus_per_node
    num_inference_gpus = args.inference_num_gpus
    total_gpus = num_training_gpus + num_inference_gpus

    if placement_strategy == "custom":
        return _create_custom_unified_placement_group(args, num_training_gpus, num_inference_gpus)

    # Single PG ensures deterministic node-to-role assignment across restarts,
    # avoiding kernel/weight cache misses from random GPU shuffling.
    logger.info(
        f"Creating unified placement group with {total_gpus} GPUs "
        f"({num_training_gpus} training + {num_inference_gpus} inference)..."
    )

    pg, sorted_bundle_indices, sorted_gpu_ids = _create_placement_group(
        total_gpus, strategy="PACK", name="unified_pg"
    )

    if placement_strategy == "training_first":
        training_bundle_indices = sorted_bundle_indices[:num_training_gpus]
        training_gpu_ids = sorted_gpu_ids[:num_training_gpus]
        inference_bundle_indices = sorted_bundle_indices[num_training_gpus:]
        inference_gpu_ids = sorted_gpu_ids[num_training_gpus:]
    else:
        inference_bundle_indices = sorted_bundle_indices[:num_inference_gpus]
        inference_gpu_ids = sorted_gpu_ids[:num_inference_gpus]
        training_bundle_indices = sorted_bundle_indices[num_inference_gpus:]
        training_gpu_ids = sorted_gpu_ids[num_inference_gpus:]

    logger.info(
        f"Placement (strategy={placement_strategy}): "
        f"training bundles={training_bundle_indices}, "
        f"inference bundles={inference_bundle_indices}"
    )

    return {
        "training": (pg, training_bundle_indices, training_gpu_ids),
        "inference": (pg, inference_bundle_indices, inference_gpu_ids),
    }


# 用已预留 PG 创建训练 actor 组；0.4 是 Ray 调度配额，不能理解为硬件隔离了 40% 显存。
def allocate_train_group(args, num_nodes, num_gpus_per_node, pg, training_class=None):
    return RayTrainGroup(
        args=args,
        num_nodes=num_nodes,
        num_gpus_per_node=num_gpus_per_node,
        pg=pg,
        num_gpus_per_actor=0.4,
        training_class=training_class,
    )


# 创建组并等待全部 actor 初始化，再校验返回状态一致；异步装配入口可直接使用 allocate+async_init。
def create_train_group(args, training_pg, training_class=None, mooncake_config=None):
    train_group = allocate_train_group(
        args=args,
        num_nodes=args.training_num_nodes,
        num_gpus_per_node=args.training_num_gpus_per_node,
        pg=training_pg,
        training_class=training_class,
    )

    some_ids = ray.get(
        train_group.async_init(
            args, role="training", mooncake_config=mooncake_config, with_ref=False
        )
    )

    assert len(set(some_ids)) == 1

    return train_group
