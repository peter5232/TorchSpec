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

# 进程组注册表：Gloo 用于控制对象/CPU 通信，USP 分别建立数据副本、序列分片及梯度同步 mesh。
# 全局 rank、组内 rank 与序列位置是不同坐标，读取时需看每个函数使用的 group。

import torch.distributed as dist

GLOO_GROUP = None

_TP_DEVICE_MESH = None
_TP_GROUP = None
_USP_DEVICE_MESH = None
_USP_GRAD_SYNC_MESH = None
_DRAFT_SP_GROUP = None
_SP_ULYSSES_GROUP = None
_SP_RING_GROUP = None


# 懒初始化覆盖所有 rank 的 Gloo 组并缓存；各参与进程必须以一致顺序创建组。
def init_gloo_group():
    """Initialize Gloo group for distributed communication."""
    global GLOO_GROUP
    if GLOO_GROUP is None:
        GLOO_GROUP = dist.new_group(backend="gloo")
    return GLOO_GROUP


# 返回已创建 Gloo 组；未初始化直接报错，避免悄悄改用默认 GPU 通信组。
def get_gloo_group():
    """Get the Gloo group for distributed communication."""
    global GLOO_GROUP
    if GLOO_GROUP is None:
        raise RuntimeError("Gloo group has not been initialized. Call _init_gloo_group() first.")
    return GLOO_GROUP


# 读取 TP 组全局引用；本文件不在此 getter 中初始化它，未设置时返回 None。
def get_tp_group():
    global _TP_GROUP
    return _TP_GROUP


# 读取 TP device mesh 引用，调用者需处理尚未设置的 None。
def get_tp_device_mesh():
    global _TP_DEVICE_MESH
    return _TP_DEVICE_MESH


# 返回二维 (draft_dp,draft_sp) mesh；无真正 SP 时可能为 None。
def get_usp_device_mesh():
    global _USP_DEVICE_MESH
    return _USP_DEVICE_MESH


# 返回覆盖全部训练 rank 的一维梯度同步 mesh，与只跨数据副本的 draft_dp mesh 区别开。
def get_usp_grad_sync_mesh():
    global _USP_GRAD_SYNC_MESH
    return _USP_GRAD_SYNC_MESH


# 以连续 SP 块分组：Ulysses 是块内连续 rank，Ring 是以 Ulysses 大小为步长的 rank。
# 例如 U=2,R=2，SP=[0,1,2,3]，Ulysses=[0,1]/[2,3]，Ring=[0,2]/[1,3]。
def _build_usp_group_ranks(
    world_size: int, sp_ulysses_size: int, sp_ring_size: int
) -> tuple[list[list[int]], list[list[int]], list[list[int]]]:
    sp_size = sp_ulysses_size * sp_ring_size
    if sp_size <= 0:
        raise ValueError(f"sp_size must be positive, got {sp_size}")
    if world_size % sp_size != 0:
        raise ValueError(
            "world_size must be divisible by sp_ulysses_size * sp_ring_size, "
            f"got world_size={world_size}, sp_ulysses_size={sp_ulysses_size}, "
            f"sp_ring_size={sp_ring_size}"
        )

    draft_sp_groups: list[list[int]] = []
    ulysses_groups: list[list[int]] = []
    ring_groups: list[list[int]] = []
    num_ulysses_pgs = sp_ring_size
    num_ring_pgs = sp_ulysses_size
    for base_rank in range(0, world_size, sp_size):
        draft_sp_groups.append(list(range(base_rank, base_rank + sp_size)))
        for idx in range(num_ulysses_pgs):
            ulysses_groups.append(
                list(
                    range(
                        base_rank + idx * sp_ulysses_size,
                        base_rank + (idx + 1) * sp_ulysses_size,
                    )
                )
            )
        for idx in range(num_ring_pgs):
            ring_groups.append(list(range(base_rank + idx, base_rank + sp_size, num_ring_pgs)))
    return draft_sp_groups, ulysses_groups, ring_groups


# 校验 world 可整除 U*R，建立二维样本/序列 mesh 与全 rank 梯度 mesh。
# Yunchang 创建具体 Ulysses/Ring 组，再验证它们不越过所属样本的 SP 组；SP=1 清空相关状态。
def init_usp_groups(sp_ulysses_size: int = 1, sp_ring_size: int = 1):
    global _USP_DEVICE_MESH
    global _USP_GRAD_SYNC_MESH
    global _DRAFT_SP_GROUP
    global _SP_ULYSSES_GROUP, _SP_RING_GROUP

    sp_size = sp_ulysses_size * sp_ring_size
    if sp_size == 1:
        _USP_DEVICE_MESH = None
        _USP_GRAD_SYNC_MESH = None
        _DRAFT_SP_GROUP = None
        _SP_ULYSSES_GROUP = None
        _SP_RING_GROUP = None
        return None, None, None

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    if world_size % sp_size != 0:
        raise ValueError(
            "world_size must be divisible by sp_ulysses_size * sp_ring_size, "
            f"got world_size={world_size}, sp_ulysses_size={sp_ulysses_size}, "
            f"sp_ring_size={sp_ring_size}"
        )

    # 例如 world=8、U=2、Ring=2，则 SP=4、独立 DP=2。
    # 二维 mesh 的一行四个 rank 共同处理一份序列，另一行处理另一份；不能把 world=8 当 batch 放大 8 倍。
    draft_dp_size = world_size // sp_size

    _DRAFT_SP_GROUP = None
    _SP_ULYSSES_GROUP = None
    _SP_RING_GROUP = None

    _USP_DEVICE_MESH = dist.device_mesh.init_device_mesh(
        "cuda",
        (draft_dp_size, sp_size),
        mesh_dim_names=("draft_dp", "draft_sp"),
    )
    _DRAFT_SP_GROUP = _USP_DEVICE_MESH.get_group("draft_sp")
    # 梯度同步单独用全 rank 一维 mesh：同一样本不同序列片段都对同一套模型参数有贡献。
    # attention 的 SP 交换与参数梯度归约不是同一个通信阶段，也不能复用相同除数的直觉。
    _USP_GRAD_SYNC_MESH = dist.device_mesh.init_device_mesh(
        "cuda",
        (world_size,),
        mesh_dim_names=("draft_dp_with_sp",),
    )

    import yunchang
    from yunchang.globals import PROCESS_GROUP as YUNCHANG_PROCESS_GROUP

    yunchang.set_seq_parallel_pg(
        sp_ulysses_degree=sp_ulysses_size,
        sp_ring_degree=sp_ring_size,
        rank=rank,
        world_size=world_size,
        # 低位 Ulysses 约定对应 rank%U 和 rank//U 的坐标分解。
        # 取数位置、RoPE position 与 Yunchang 通信布局必须一致，后面的成员校验用于发现越组配置。
        use_ulysses_low=True,
    )
    _SP_ULYSSES_GROUP = YUNCHANG_PROCESS_GROUP.ULYSSES_PG
    _SP_RING_GROUP = YUNCHANG_PROCESS_GROUP.RING_PG
    _validate_usp_group_composition()

    return _DRAFT_SP_GROUP, _SP_ULYSSES_GROUP, _SP_RING_GROUP


# 返回共同处理一条样本的 SP 组，用于读取分片坐标及组内通信。
def get_draft_sp_group():
    global _DRAFT_SP_GROUP
    return _DRAFT_SP_GROUP


# 返回 Ulysses 注意力通信组；其成员在当前低位 Ulysses 布局中连续。
def get_sp_ulysses_group():
    global _SP_ULYSSES_GROUP
    return _SP_ULYSSES_GROUP


# 返回 Ring 注意力通信组；同一 Ring 组连接不同 Ring 块的相同 Ulysses 坐标。
def get_sp_ring_group():
    global _SP_RING_GROUP
    return _SP_RING_GROUP


# 返回本 rank 在完整 SP 组中的索引，无 SP 组时为 0。
def get_sp_rank() -> int:
    sp_group = get_draft_sp_group()
    if sp_group is None:
        return 0
    return dist.get_rank(sp_group)


# 返回 Ulysses 组内索引，无该组时为 0，不等同于全局 rank。
def get_sp_ulysses_rank() -> int:
    ulysses_group = get_sp_ulysses_group()
    if ulysses_group is None:
        return 0
    return dist.get_rank(ulysses_group)


# 返回 Ring 组内索引，无该组时为 0，供计算本 Ring 块的绝对位置。
def get_sp_ring_rank() -> int:
    ring_group = get_sp_ring_group()
    if ring_group is None:
        return 0
    return dist.get_rank(ring_group)


# 把 sp_rank 分解为 (rank%U,rank//U)，先校验范围。
# 例如 U=2,R=3，sp_rank=5 对应 Ulysses=1、Ring=2。
def get_usp_rank_coords(sp_rank: int, sp_ulysses_size: int, sp_ring_size: int) -> tuple[int, int]:
    sp_size = sp_ulysses_size * sp_ring_size
    if sp_rank < 0 or sp_rank >= sp_size:
        raise ValueError(f"sp_rank must be in [0, {sp_size}), got {sp_rank}")
    ulysses_rank = sp_rank % sp_ulysses_size
    ring_rank = sp_rank // sp_ulysses_size
    return ulysses_rank, ring_rank


# 在指定组内收集各成员的全局 rank，返回 tuple，供验证实际通信拓扑。
def _gather_group_members(group) -> tuple[int, ...]:
    group_world_size = dist.get_world_size(group)
    members = [None] * group_world_size
    dist.all_gather_object(members, dist.get_rank(), group=group)
    return tuple(members)


# 收集 SP/Ring/Ulysses 实际成员，逐记录检查 Ring/Ulysses 都属于当前 SP 样本组。
# 该检查防止 attention 在不同训练样本之间错误交换特征。
def _validate_usp_group_composition() -> None:
    sp_group = get_draft_sp_group()
    ulysses_group = get_sp_ulysses_group()
    ring_group = get_sp_ring_group()
    if sp_group is None or ulysses_group is None or ring_group is None:
        raise RuntimeError("USP groups must be initialized before validating group composition")

    sp_members = _gather_group_members(sp_group)
    local_record = {
        "world_rank": dist.get_rank(),
        "ring_members": _gather_group_members(ring_group),
        "ulysses_members": _gather_group_members(ulysses_group),
    }
    records = [None] * dist.get_world_size(sp_group)
    dist.all_gather_object(records, local_record, group=sp_group)
    for record in records:
        ring_members = tuple(record["ring_members"])
        ulysses_members = tuple(record["ulysses_members"])
        if any(member not in sp_members for member in ring_members + ulysses_members):
            raise RuntimeError("USP ring/ulysses groups include ranks outside the draft SP group")
