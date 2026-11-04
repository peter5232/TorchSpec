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

# 根据推理后端和 Ray placement group 创建 actor，并把慢速模型初始化发到目标节点。
# prepare 返回尚待等待的 ObjectRef，阻塞入口则等待就绪；多节点只把 head 返回给请求分发器。

"""Inference engine creation and initialization with Ray placement groups."""

import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from torchspec.utils.env import get_torchspec_env_vars
from torchspec.utils.logging import logger

# Multi-node TP worker engines must stay alive to participate in NCCL
# operations but are never called directly. Store refs here to prevent GC.
_alive_worker_engines: list = []


# 阻塞式总入口：离线回放单独用 CPU 路径，在线后端按类型建立引擎并等到初始化结束。
def create_inference_engines(args, inference_pg, mooncake_config, engine_group: int = 0):
    """Create inference engines based on configured engine type (blocking).

    Supports live inference engines and the CPU-only ``offline`` replay engine.

    Returns:
        List of head engines used for dispatching requests. Multi-node TP
        worker engines (if any) are kept alive internally but not returned.
    """
    engine_type = getattr(args, "inference_engine_type", "hf")
    if engine_type == "offline":
        engines, init_refs = _prepare_offline_replay_engines(args, mooncake_config, engine_group)
        _wait_for_init(init_refs, "OfflineReplay", timeout=300)
        return engines

    if engine_type not in ("hf", "sgl", "vllm", "trtllm", "tokenspeed"):
        raise ValueError(f"Unknown inference_engine_type: {engine_type}")

    logger.info(f"Using {engine_type} engine for inference")

    engines = init_engines(
        args,
        inference_pg,
        engine_type=engine_type,
        mooncake_config=mooncake_config,
        engine_group=engine_group,
    )

    logger.info(
        f"Created {len(engines)} distributed {engine_type} engines from {args.target_model_path}"
    )
    return engines


# 只创建 actor 和发送 init，返回引擎及初始化引用；上层可同时准备训练侧，使用前必须等待引用完成。
def prepare_inference_engines(args, inference_pg, mooncake_config, engine_group: int = 0):
    """Create inference engines and fire init calls without waiting.

    Use this to parallelize engine initialization with other setup work
    (e.g., training actor initialization). Call ray.get() on the returned
    init_refs before using the engines.

    Returns:
        Tuple of (head_engines, init_refs) where head_engines are the engines
        for dispatching requests, and init_refs are ObjectRefs to wait on.
    """
    engine_type = getattr(args, "inference_engine_type", "hf")
    if engine_type == "offline":
        return _prepare_offline_replay_engines(args, mooncake_config, engine_group)

    if engine_type not in ("hf", "sgl", "vllm", "trtllm", "tokenspeed"):
        raise ValueError(f"Unknown inference_engine_type: {engine_type}")

    logger.info(f"Preparing {engine_type} inference engines...")

    if engine_type == "hf":
        engines, init_refs = _prepare_hf_engines(args, inference_pg, mooncake_config, engine_group)
    elif engine_type == "sgl":
        engines, init_refs = _prepare_sgl_engines(args, inference_pg, mooncake_config, engine_group)
    elif engine_type == "trtllm":
        engines, init_refs = _prepare_trtllm_engines(
            args, inference_pg, mooncake_config, engine_group
        )
    elif engine_type == "tokenspeed":
        engines, init_refs = _prepare_tokenspeed_engines(
            args, inference_pg, mooncake_config, engine_group
        )
    else:
        engines, init_refs = _prepare_vllm_engines(
            args, inference_pg, mooncake_config, engine_group
        )

    return engines, init_refs


# 创建正数个 CPU 回放 actor，不占推理 placement GPU；每个 actor 都收到相同 Mooncake 配置。
def _prepare_offline_replay_engines(
    args, mooncake_config, engine_group: int = 0
) -> tuple[list, list]:
    """Create CPU replay actors; they do not consume inference placement GPUs."""
    from torchspec.inference.engine.offline_replay_engine import OfflineReplayEngine

    num_engines = getattr(args, "offline_num_engines", 1)
    if num_engines <= 0:
        raise ValueError("inference.offline.num_engines must be positive")
    ReplayRayActor = ray.remote(OfflineReplayEngine)
    env_vars = get_torchspec_env_vars()
    engines = [
        ReplayRayActor.options(
            num_cpus=1,
            num_gpus=0,
            runtime_env={"env_vars": env_vars},
        ).remote(args=args, rank=rank, engine_group=engine_group)
        for rank in range(num_engines)
    ]
    init_refs = [engine.init.remote(mooncake_config=mooncake_config) for engine in engines]
    logger.info("Preparing %d CPU offline replay engine(s)", num_engines)
    return engines, init_refs


# 把已选择的后端路由到阻塞式初始化器；未知类型直接报错，不隐式换成其他引擎。
def init_engines(args, pg, engine_type: str, mooncake_config=None, engine_group: int = 0) -> list:
    """Initialize inference engines with Ray placement groups.

    Args:
        args: Configuration arguments.
        pg: Placement group tuple (pg, reordered_bundle_indices, reordered_gpu_ids).
        engine_type: Engine type ("hf", "sgl", "vllm", or "trtllm").
        mooncake_config: MooncakeConfig object.

    Returns:
        List of head engines for dispatching requests.
    """
    if engine_type == "hf":
        return _init_hf_engines(args, pg, mooncake_config, engine_group)
    elif engine_type == "sgl":
        return _init_sgl_engines(args, pg, mooncake_config, engine_group)
    elif engine_type == "vllm":
        return _init_vllm_engines(args, pg, mooncake_config, engine_group)
    elif engine_type == "trtllm":
        return _init_trtllm_engines(args, pg, mooncake_config, engine_group)
    elif engine_type == "tokenspeed":
        return _init_tokenspeed_engines(args, pg, mooncake_config, engine_group)
    else:
        raise ValueError(f"Unknown engine_type: {engine_type}")


# 按单节点可容纳的 GPU 数限制每引擎配置，再通过公共 actor 工厂启动 HF 模型加载。
def _prepare_hf_engines(args, pg, mooncake_config=None, engine_group: int = 0) -> tuple[list, list]:
    """Create HF engine actors and fire init calls without waiting.

    Returns:
        Tuple of (engines, init_handles).
    """
    num_gpus_total = getattr(args, "inference_num_gpus", 1)
    num_gpus_per_engine = getattr(args, "inference_num_gpus_per_engine", 1)
    num_gpus_per_node = getattr(args, "inference_num_gpus_per_node", 8)

    num_gpus_per_engine = min(num_gpus_per_engine, num_gpus_per_node)
    num_engines = num_gpus_total // num_gpus_per_engine

    logger.info(f"Initializing {num_engines} HF engines ({num_gpus_per_engine} GPU(s) each)")

    from torchspec.inference.engine.hf_engine import HFEngine

    HFRayActor = ray.remote(HFEngine)
    return _create_and_init_actors(
        args,
        pg,
        num_engines,
        num_gpus_per_engine,
        HFRayActor,
        mooncake_config,
        engine_group=engine_group,
    )


# 等待 HF 初始化引用，采用较长模型加载超时后返回可调用引擎。
def _init_hf_engines(args, pg, mooncake_config=None, engine_group: int = 0) -> list:
    """Initialize HF engines with Ray placement groups."""
    engines, init_handles = _prepare_hf_engines(args, pg, mooncake_config, engine_group)
    _wait_for_init(init_handles, "HF", timeout=1200)
    return engines


# 先创建所有节点 actor，再为每个模型副本协商地址、逐个选端口，最后并发触发初始化。
# 跨节点 worker 保留在模块列表以免被回收；只有 node_rank=0 的 head 接收 generate。
def _prepare_sgl_engines(
    args, pg, mooncake_config=None, engine_group: int = 0
) -> tuple[list, list]:
    """Create SGL engine actors and fire init calls without waiting.

    Handles three cases:
      - Single-node, multiple engines: one engine per group of GPUs
      - Multi-node, single replica: one engine per node, all forming one TP group
      - Multi-node, multiple replicas: N independent TP groups, each spanning nnodes

    For multi-node, worker engines are stored in a module-level list to prevent GC.

    Returns:
        Tuple of (head_engines, init_handles). head_engines are the engines that
        accept generate() calls. init_handles are ObjectRefs for ALL engines
        (head + worker) that must be waited on before use.
    """

    nnodes = getattr(args, "sglang_nnodes", 1)
    num_gpus_total = getattr(args, "inference_num_gpus", 1)

    if nnodes > 1:
        gpus_per_node = getattr(args, "inference_num_gpus_per_node", 8)
        # 例如总 32 卡、每节点 8 卡、每副本跨 2 节点：得到 2 个副本、4 个 actor。
        # 最终只有 2 个 head 接请求，其余 actor 仍必须存活以参与同一副本的 TP 通信。
        gpus_per_replica = nnodes * gpus_per_node
        num_replicas = num_gpus_total // gpus_per_replica
        num_engines = num_replicas * nnodes
        gpus_per_engine = gpus_per_node
    else:
        gpus_per_engine = getattr(args, "inference_num_gpus_per_engine", 1)
        num_replicas = num_gpus_total // gpus_per_engine
        num_engines = num_replicas

    logger.info(
        f"Initializing {num_engines} Sgl engines "
        f"({gpus_per_engine} GPU(s) each, nnodes={nnodes}, replicas={num_replicas})"
    )

    from torchspec.inference.engine.sgl_engine import SglEngine

    # placement group 负责资源位置；重排后的 bundle 和 GPU ID 必须使用同一 offset，
    # 否则 actor 所在节点与它准备绑定的设备会不一致。
    pg_obj, reordered_bundle_indices, reordered_gpu_ids = pg

    SglRayActor = ray.remote(SglEngine)
    env_vars = get_torchspec_env_vars()

    # Step 1: Create all engine actors (without calling init yet)
    engines = []
    for i in range(num_engines):
        # 连续 nnodes 个 actor 属于同一模型副本，模运算得到其副本内节点号。
        node_rank = i % nnodes if nnodes > 1 else 0

        bundle_offset = i * gpus_per_engine
        base_gpu_id = int(reordered_gpu_ids[bundle_offset])

        scheduling_strategy = PlacementGroupSchedulingStrategy(
            placement_group=pg_obj,
            placement_group_capture_child_tasks=True,
            placement_group_bundle_index=reordered_bundle_indices[bundle_offset],
        )

        engine = SglRayActor.options(
            num_cpus=0.2,
            num_gpus=0.2,
            scheduling_strategy=scheduling_strategy,
            runtime_env={"env_vars": env_vars},
        ).remote(
            args=args,
            rank=i,
            base_gpu_id=base_gpu_id,
            num_gpus_per_engine=gpus_per_engine,
            node_rank=node_rank,
            engine_group=engine_group,
        )
        engines.append(engine)

    # Step 2: Auto-negotiate dist_init_addr for each multi-node replica
    # Uses RayActor.get_node_ip() and .find_free_port() on the node_rank=0 engine
    # 每份模型副本建立独立通信地址；两个副本不能误连进同一个 TP 进程组。
    dist_init_addrs: dict[int, str] = {}
    if nnodes > 1:
        configured_addr = getattr(args, "sglang_dist_init_addr", None)
        for replica_idx in range(num_replicas):
            if configured_addr and num_replicas == 1:
                dist_init_addrs[replica_idx] = configured_addr
                logger.info(
                    f"Replica {replica_idx}: using configured dist_init_addr: {configured_addr}"
                )
            else:
                head_engine = engines[replica_idx * nnodes]
                ip, port = ray.get(
                    [head_engine.get_node_ip.remote(), head_engine.find_free_port.remote()],
                    timeout=120,
                )
                addr = f"{ip}:{port}"
                dist_init_addrs[replica_idx] = addr
                logger.info(f"Replica {replica_idx}: auto-negotiated dist_init_addr: {addr}")

    # Step 2.5: Pre-allocate ports to avoid TOCTOU races between parallel engines.
    # Each engine needs 2 consecutive ports; allocate sequentially so engines
    # on the same node never collide.
    # 串行寻找端口降低并发初始化时选到相同端口的风险；扫描并不等同永久绑定端口。
    pre_allocated_ports: dict[int, int] = {}
    next_start = 10000
    for i in range(num_engines):
        port = ray.get(
            engines[i].find_free_port.remote(start_port=next_start, consecutive=2),
            timeout=120,
        )
        pre_allocated_ports[i] = port
        next_start = port + 2
        logger.info(f"Engine {i}: pre-allocated ports {port}, {port + 1}")

    # Step 3: Fire init() on all engines (non-blocking)
    # 所有远程初始化先发出去再统一等待，允许不同节点并行加载。
    # 若逐个 init 后立刻等结果，多节点集合通信可能一直等尚未启动的同伴。
    init_handles = []
    for i, engine in enumerate(engines):
        replica_idx = i // nnodes if nnodes > 1 else i
        init_handles.append(
            engine.init.remote(
                mooncake_config=mooncake_config,
                dist_init_addr=dist_init_addrs.get(replica_idx),
                pre_allocated_port=pre_allocated_ports.get(i),
            )
        )

    # Separate head/worker engines for multi-node
    if nnodes > 1:
        head_engines = [engines[i] for i in range(num_engines) if i % nnodes == 0]
        worker_engines = [engines[i] for i in range(num_engines) if i % nnodes != 0]
        # Keep worker engines alive (they participate in NCCL TP) via module-level ref
        _alive_worker_engines.extend(worker_engines)
        logger.info(
            f"Prepared multi-node Sgl engines: {len(head_engines)} heads + "
            f"{len(worker_engines)} workers ({num_replicas} replicas)"
        )
        return head_engines, init_handles

    return engines, init_handles


# 等待 head 和 worker 全部完成 SGLang 初始化；超时依据配置及单/多节点默认值决定。
def _init_sgl_engines(args, pg, mooncake_config=None, engine_group: int = 0) -> list:
    """Initialize SGLang engines with Ray placement groups (blocking)."""
    head_engines, init_handles = _prepare_sgl_engines(args, pg, mooncake_config, engine_group)
    nnodes = getattr(args, "sglang_nnodes", 1)
    init_timeout = getattr(args, "sglang_init_timeout", 300 if nnodes == 1 else 600)
    _wait_for_init(init_handles, "Sgl", timeout=init_timeout)
    return head_engines


# 按副本和节点划分 actor，每个副本使用自己的 head 地址与预选端口，初始化引用覆盖所有节点。
# 返回的 head 数等于副本数，不等于参与 TP/PP 的 actor 总数。
def _prepare_vllm_engines(
    args, pg, mooncake_config=None, engine_group: int = 0
) -> tuple[list, list]:
    """Create vLLM engine actors and fire init calls without waiting.

    Handles three cases:
      - Single-node, multiple engines: one engine per group of GPUs
      - Multi-node, single replica: one engine per node, all forming one TP group
      - Multi-node, multiple replicas: N independent TP groups, each spanning nnodes

    For multi-node, worker engines are stored in a module-level list to prevent GC.

    Returns:
        Tuple of (head_engines, init_handles). head_engines are the engines that
        accept generate() calls. init_handles are ObjectRefs for ALL engines
        (head + worker) that must be waited on before use.
    """
    nnodes = getattr(args, "vllm_nnodes", 1)
    num_gpus_total = getattr(args, "inference_num_gpus", 1)

    # 先按一份副本所需节点数计算副本数量；跨节点 actor 的数量不是可独立服务的模型数量。
    if nnodes > 1:
        gpus_per_node = getattr(args, "inference_num_gpus_per_node", 8)
        gpus_per_replica = nnodes * gpus_per_node
        num_replicas = num_gpus_total // gpus_per_replica
        num_engines = num_replicas * nnodes
        gpus_per_engine = gpus_per_node
    else:
        gpus_per_engine = getattr(args, "inference_num_gpus_per_engine", 1)
        num_replicas = num_gpus_total // gpus_per_engine
        num_engines = num_replicas

    logger.info(
        f"Initializing {num_engines} vLLM engines "
        f"({gpus_per_engine} GPU(s) each, nnodes={nnodes}, replicas={num_replicas})"
    )

    from torchspec.inference.engine.vllm_engine import VllmEngine

    pg_obj, reordered_bundle_indices, reordered_gpu_ids = pg

    VllmRayActor = ray.remote(VllmEngine)
    env_vars = get_torchspec_env_vars()

    engines = []
    for i in range(num_engines):
        # 副本内 rank 0 拥有请求前端，其他节点由 VllmEngine 启动 headless executor。
        node_rank = i % nnodes if nnodes > 1 else 0

        bundle_offset = i * gpus_per_engine
        base_gpu_id = int(reordered_gpu_ids[bundle_offset])

        scheduling_strategy = PlacementGroupSchedulingStrategy(
            placement_group=pg_obj,
            placement_group_capture_child_tasks=True,
            placement_group_bundle_index=reordered_bundle_indices[bundle_offset],
        )

        engine = VllmRayActor.options(
            num_cpus=0.2,
            num_gpus=0.2,
            scheduling_strategy=scheduling_strategy,
            runtime_env={"env_vars": env_vars},
        ).remote(
            args=args,
            rank=i,
            base_gpu_id=base_gpu_id,
            num_gpus_per_engine=gpus_per_engine,
            node_rank=node_rank,
            engine_group=engine_group,
        )
        engines.append(engine)

    # Pre-allocate ports to avoid TOCTOU races between parallel engines.
    # Each engine needs 2 consecutive ports (dist_init + nccl).
    # 先固定每个 actor 的端口候选，再从 head 的候选构造副本初始化地址，保持双方配置一致。
    pre_allocated_ports: dict[int, int] = {}
    next_start = 10000
    for i in range(num_engines):
        port = ray.get(
            engines[i].find_free_port.remote(start_port=next_start, consecutive=2),
            timeout=30,
        )
        pre_allocated_ports[i] = port
        next_start = port + 2
        logger.info(f"vLLM Engine {i}: pre-allocated ports {port}, {port + 1}")

    dist_init_addrs: dict[int, str] = {}
    if nnodes > 1:
        configured_addr = getattr(args, "vllm_dist_init_addr", None)
        for replica_idx in range(num_replicas):
            if configured_addr and num_replicas == 1:
                dist_init_addrs[replica_idx] = configured_addr
                logger.info(
                    f"Replica {replica_idx}: using configured dist_init_addr: {configured_addr}"
                )
            else:
                head_idx = replica_idx * nnodes
                head_engine = engines[head_idx]
                ip = ray.get(head_engine.get_node_ip.remote(), timeout=30)
                addr = f"{ip}:{pre_allocated_ports[head_idx]}"
                dist_init_addrs[replica_idx] = addr
                logger.info(
                    f"Replica {replica_idx}: dist_init_addr from pre-allocated port: {addr}"
                )

    # 返回值保留所有节点的初始化引用；只等待 head 不足以证明 follower 已进入通信组。
    init_handles = []
    for i, engine in enumerate(engines):
        replica_idx = i // nnodes if nnodes > 1 else i
        init_handles.append(
            engine.init.remote(
                mooncake_config=mooncake_config,
                dist_init_addr=dist_init_addrs.get(replica_idx),
                pre_allocated_port=pre_allocated_ports.get(i),
            )
        )

    if nnodes > 1:
        head_engines = [engines[i] for i in range(num_engines) if i % nnodes == 0]
        worker_engines = [engines[i] for i in range(num_engines) if i % nnodes != 0]
        # 保存非请求入口的强引用，避免局部列表退出作用域后 worker actor 被回收。
        _alive_worker_engines.extend(worker_engines)
        logger.info(
            f"Prepared multi-node vLLM engines: {len(head_engines)} heads + "
            f"{len(worker_engines)} workers ({num_replicas} replicas)"
        )
        return head_engines, init_handles

    return engines, init_handles


# 同步等待全部 vLLM 节点就绪，再只返回请求入口列表。
def _init_vllm_engines(args, pg, mooncake_config=None, engine_group: int = 0) -> list:
    """Initialize vLLM engines with Ray placement groups (blocking)."""
    head_engines, init_handles = _prepare_vllm_engines(args, pg, mooncake_config, engine_group)
    nnodes = getattr(args, "vllm_nnodes", 1)
    init_timeout = getattr(args, "vllm_init_timeout", 300 if nnodes == 1 else 600)
    _wait_for_init(init_handles, "Vllm", timeout=init_timeout)
    return head_engines


# 为每个单节点 TP 副本分配连续 GPU；TP=1 由 Ray 限制可见设备，TP>1 由后端显式绑定设备块。
def _prepare_trtllm_engines(
    args, pg, mooncake_config=None, engine_group: int = 0
) -> tuple[list, list]:
    """Create TensorRT-LLM engine actors and fire init calls without waiting.

    Single-node only: each engine owns a contiguous block of
    ``inference_num_gpus_per_engine`` GPUs and runs TRT-LLM's own MPI workers
    across them (TP degree = num_gpus_per_engine). Unlike sgl/vllm there is no
    cross-node dist-init negotiation or port pre-allocation — TRT-LLM manages
    worker bring-up internally.

    Returns:
        Tuple of (engines, init_handles).
    """
    nnodes = getattr(args, "trtllm_nnodes", 1)
    if nnodes > 1:
        raise NotImplementedError(
            f"trtllm backend supports single-node TP only (trtllm_nnodes={nnodes})."
        )

    num_gpus_total = getattr(args, "inference_num_gpus", 1)
    gpus_per_engine = getattr(args, "inference_num_gpus_per_engine", 1)
    num_engines = num_gpus_total // gpus_per_engine

    logger.info(
        f"Initializing {num_engines} TensorRT-LLM engines "
        f"({gpus_per_engine} GPU(s) each, tp_size={gpus_per_engine})"
    )

    from torchspec.inference.engine.trtllm_engine import TrtllmEngine

    pg_obj, reordered_bundle_indices, reordered_gpu_ids = pg

    TrtllmRayActor = ray.remote(TrtllmEngine)
    env_vars = get_torchspec_env_vars()

    # TRT-LLM's MPI workers bind GPUs themselves, so assignment differs by TP degree:
    # tp==1 drops NOSET and reserves a whole GPU so Ray scopes it as cuda:0 per engine;
    # tp>1 keeps NOSET, pins a contiguous block in init(), and reserves a 0.2 placeholder.
    # 单 GPU actor 让 Ray 设置可见设备；多 GPU TP 则让 MPI worker 看到引擎专属的连续设备块。
    single_gpu = gpus_per_engine == 1
    if single_gpu:
        env_vars.pop("RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES", None)
    actor_num_gpus = float(gpus_per_engine) if single_gpu else 0.2

    engines = []
    for i in range(num_engines):
        # 每个副本按其卡数跳过前一块资源，而不是让相邻副本都从 GPU 0 开始。
        bundle_offset = i * gpus_per_engine
        base_gpu_id = int(reordered_gpu_ids[bundle_offset])

        scheduling_strategy = PlacementGroupSchedulingStrategy(
            placement_group=pg_obj,
            placement_group_capture_child_tasks=True,
            placement_group_bundle_index=reordered_bundle_indices[bundle_offset],
        )

        engine = TrtllmRayActor.options(
            num_cpus=0.2,
            num_gpus=actor_num_gpus,
            scheduling_strategy=scheduling_strategy,
            runtime_env={"env_vars": env_vars},
        ).remote(
            args=args,
            rank=i,
            base_gpu_id=base_gpu_id,
            num_gpus_per_engine=gpus_per_engine,
            node_rank=0,
            engine_group=engine_group,
        )
        engines.append(engine)

    # TRT-LLM 自行管理 MPI 启动，无需复制 SGLang/vLLM 的跨节点地址协商；仍汇合初始化引用。
    init_handles = [engine.init.remote(mooncake_config=mooncake_config) for engine in engines]
    return engines, init_handles


# 等待 TRT-LLM 自身 MPI worker 初始化完成，超时使用该后端配置。
def _init_trtllm_engines(args, pg, mooncake_config=None, engine_group: int = 0) -> list:
    """Initialize TensorRT-LLM engines with Ray placement groups (blocking)."""
    engines, init_handles = _prepare_trtllm_engines(args, pg, mooncake_config, engine_group)
    init_timeout = getattr(args, "trtllm_init_timeout", 600)
    _wait_for_init(init_handles, "Trtllm", timeout=init_timeout)
    return engines


# 仅接受单节点且至少一个引擎，把连续 GPU 块和 placement bundle 传给 TokenSpeed actor。
def _prepare_tokenspeed_engines(
    args, pg, mooncake_config=None, engine_group: int = 0
) -> tuple[list, list]:
    """Create single-node TokenSpeed engine actors."""
    nnodes = getattr(args, "tokenspeed_nnodes", 1)
    if nnodes != 1:
        raise NotImplementedError(
            "The initial TokenSpeed backend supports single-node inference only"
        )

    num_gpus_total = getattr(args, "inference_num_gpus", 1)
    gpus_per_engine = getattr(args, "inference_num_gpus_per_engine", 1)
    # 模型副本数由总卡数整除每副本卡数得到；零副本是配置错误，不能返回空列表悄悄继续。
    num_engines = num_gpus_total // gpus_per_engine
    if num_engines <= 0:
        raise ValueError("TokenSpeed requires at least one inference engine")

    from torchspec.inference.engine.tokenspeed_engine import TokenSpeedEngine

    pg_obj, reordered_bundle_indices, reordered_gpu_ids = pg
    TokenSpeedRayActor = ray.remote(TokenSpeedEngine)
    env_vars = get_torchspec_env_vars()
    engines = []
    for index in range(num_engines):
        # 用同一偏移定位 placement bundle 和第一张物理 GPU，使设备连续块与调度位置对应。
        bundle_offset = index * gpus_per_engine
        base_gpu_id = int(reordered_gpu_ids[bundle_offset])
        scheduling_strategy = PlacementGroupSchedulingStrategy(
            placement_group=pg_obj,
            placement_group_capture_child_tasks=True,
            placement_group_bundle_index=reordered_bundle_indices[bundle_offset],
        )
        engines.append(
            TokenSpeedRayActor.options(
                num_cpus=0.2,
                num_gpus=0.2,
                scheduling_strategy=scheduling_strategy,
                runtime_env={"env_vars": env_vars},
            ).remote(
                args=args,
                rank=index,
                base_gpu_id=base_gpu_id,
                num_gpus_per_engine=gpus_per_engine,
                node_rank=0,
                engine_group=engine_group,
            )
        )

    # 引擎对象先全部创建，模型加载请求随后发送，返回 init_refs 让调用者选择何时同步。
    init_refs = [engine.init.remote(mooncake_config=mooncake_config) for engine in engines]
    logger.info(
        "Preparing %d TokenSpeed engine(s), %d GPU(s) each",
        num_engines,
        gpus_per_engine,
    )
    return engines, init_refs


# 在非阻塞准备之后等待所有 TokenSpeed init 引用，避免尚未加载模型就分发请求。
def _init_tokenspeed_engines(args, pg, mooncake_config=None, engine_group: int = 0) -> list:
    engines, init_refs = _prepare_tokenspeed_engines(args, pg, mooncake_config, engine_group)
    timeout = getattr(args, "tokenspeed_init_timeout", 600)
    _wait_for_init(init_refs, "TokenSpeed", timeout=timeout)
    return engines


# 公共构建器将重排后的 bundle 与物理 GPU 基址配对，传入环境变量，再发送每个 actor 的 init 调用。
# num_gpus=0.2 是 actor 的调度占位；实际模型设备范围来自 placement 规划和后端配置，不能解释为模型只用五分之一张卡。
def _create_and_init_actors(
    args,
    pg,
    num_engines,
    num_gpus_per_engine,
    ray_actor_cls,
    mooncake_config,
    extra_kwargs=None,
    engine_group: int = 0,
) -> tuple[list, list]:
    """Create Ray actors and start their init calls.

    Args:
        args: Configuration arguments.
        pg: Placement group tuple.
        num_engines: Number of engines to create.
        num_gpus_per_engine: GPUs per engine.
        ray_actor_cls: The ray.remote-wrapped actor class.
        mooncake_config: Mooncake configuration dict.
        extra_kwargs: Extra kwargs to pass to actor constructor (e.g. num_gpus_per_engine for Sgl).
        engine_group: Group index for log file disambiguation.

    Returns:
        Tuple of (engines list, init_handles list).
    """
    pg_obj, reordered_bundle_indices, reordered_gpu_ids = pg

    engines = []
    init_handles = []

    for i in range(num_engines):
        num_gpus = 0.2
        num_cpus = num_gpus

        base_gpu_id = int(reordered_gpu_ids[i * num_gpus_per_engine])

        # 指定 bundle 固定 actor 放置位置，并让子任务继承 placement group 约束。
        scheduling_strategy = PlacementGroupSchedulingStrategy(
            placement_group=pg_obj,
            placement_group_capture_child_tasks=True,
            placement_group_bundle_index=reordered_bundle_indices[i * num_gpus_per_engine],
        )

        env_vars = get_torchspec_env_vars()

        # 构造参数只携带配置和部署身份，模型初始化留到远端 init；额外参数可补充后端专用信息。
        constructor_kwargs = {
            "args": args,
            "rank": i,
            "base_gpu_id": base_gpu_id,
            "engine_group": engine_group,
        }
        if extra_kwargs:
            constructor_kwargs.update(extra_kwargs)

        engine = ray_actor_cls.options(
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            scheduling_strategy=scheduling_strategy,
            runtime_env={"env_vars": env_vars},
        ).remote(**constructor_kwargs)

        engines.append(engine)
        # 把未完成初始化的引用保存下来，后续由统一等待函数处理超时和异常。
        init_handles.append(engine.init.remote(mooncake_config=mooncake_config))

    return engines, init_handles


# 用 ray.get 汇合初始化任务；超时后仅查询哪些引用完成以补充日志，并重新抛出原错误。
def _wait_for_init(init_handles: list, engine_label: str, timeout: int = 300):
    """Wait for engine init calls to complete with error handling.

    Args:
        init_handles: List of init() call ObjectRefs.
        engine_label: Label for log messages (e.g. "HF", "Sgl").
        timeout: Timeout in seconds.
    """
    try:
        logger.info(
            f"Waiting for {len(init_handles)} {engine_label} engines to initialize "
            f"(timeout={timeout}s)..."
        )
        ray.get(init_handles, timeout=timeout)
    except ray.exceptions.GetTimeoutError:
        logger.error(f"Timeout waiting for {engine_label} engines to initialize after {timeout}s")
        ready, not_ready = ray.wait(init_handles, num_returns=len(init_handles), timeout=0)
        logger.error(f"Initialized: {len(ready)}/{len(init_handles)}, Failed: {len(not_ready)}")
        raise
    except Exception as e:
        logger.error(f"Failed to initialize {engine_label} engines: {e}")
        raise

    logger.info(f"Successfully initialized {len(init_handles)} {engine_label} engines")
