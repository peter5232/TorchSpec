# 离线回放把磁盘已有 teacher 张量重新发布到 Mooncake，复用在线训练的消费契约。
# 它不运行 teacher 模型；工厂给它分配 CPU actor，并关闭 GPU Direct。
"""Inference-engine adapter that replays materialized target tensors from disk."""

from __future__ import annotations

import dataclasses
import uuid
from typing import Any

import ray
import torch

from torchspec.inference.engine.base import InferenceEngine
from torchspec.offline.dataset import OfflineDataset, configure_offline_args
from torchspec.ray.ray_actor import RayActor
from torchspec.transfer.mooncake.eagle_store import EagleMooncakeStore
from torchspec.utils.logging import logger, setup_file_logging


# 保留与在线引擎相同的 generate 接口，使异步编排无需为磁盘数据另建消费路径。
class OfflineReplayEngine(InferenceEngine, RayActor):
    """Serve recorded target-model outputs through the normal Mooncake contract.

    The async inference manager treats this actor like any other inference
    engine. Instead of running a target model, ``generate`` looks records up by
    data ID, writes their tensors to Mooncake, and returns the same metadata as
    a live engine.
    """

    # 初始化数据集、存储及按 data_id 检索的索引占位；此时尚未读取记录正文。
    def __init__(self, args, rank: int, engine_group: int = 0, **_kwargs) -> None:
        self.args = args
        self.rank = rank
        self._mooncake_config = None
        self._mooncake_store: EagleMooncakeStore | None = None
        self._dataset: OfflineDataset | None = None
        self._rows_by_id: dict[str, dict[str, Any]] = {}
        setup_file_logging("offline_replay", rank, group=engine_group)

    # 复制配置并禁用 GPU Direct，读取 train/eval 索引且拒绝跨 split 重复 ID，再建立 CPU 存储客户端。
    def init(self, mooncake_config=None) -> None:
        if mooncake_config is None:
            raise ValueError("OfflineReplayEngine requires a Mooncake configuration")

        from torchspec.transfer.mooncake.utils import check_mooncake_master_available

        mooncake_config = dataclasses.replace(
            mooncake_config,
            local_hostname=self.get_node_ip(),
            enable_gpu_direct=False,
        )
        check_mooncake_master_available(
            mooncake_config.master_server_address,
            mooncake_config.metadata_server,
        )
        self._mooncake_config = mooncake_config
        self._dataset = OfflineDataset(self.args.offline_data_path)
        configure_offline_args(self._dataset, self.args)
        rows = self._dataset.rows("train") + self._dataset.rows("eval")
        # 这里是启动时的索引快照；之后别的进程继续 append manifest，本 actor 不会自动发现新 ID。
        # 控制器也从 manifest 建立训练条目，所以正常流程是先完成物化，再启动回放训练。
        self._rows_by_id = {str(row["data_id"]): row for row in rows}
        if len(self._rows_by_id) != len(rows):
            raise ValueError("Offline dataset contains duplicate data IDs across splits")

        self._mooncake_store = EagleMooncakeStore(mooncake_config)
        self._mooncake_store.setup(device=torch.device("cpu"))
        logger.info(
            "OfflineReplayEngine rank %d initialized with %d records from %s",
            self.rank,
            len(rows),
            self.args.offline_data_path,
        )

    # 按 ID 定位磁盘记录，为本次发布生成新键，写入四种可选张量并带回原 mask 和元数据。
    def _replay_one(self, data_id: str) -> dict[str, Any]:
        if self._dataset is None or self._mooncake_store is None:
            raise RuntimeError("OfflineReplayEngine not initialized. Call init() first.")
        row = self._rows_by_id.get(str(data_id))
        if row is None:
            raise KeyError(f"Offline dataset has no record for data_id={data_id!r}")
        record = self._dataset.load(row)
        # data_id 是稳定样本身份，key 是一次传输的身份：同一 "train-1" 在两个 epoch
        # 会得到不同 key，避免上一轮消费者删除临时对象时误删下一轮已发布的同名对象。
        key = str(uuid.uuid4())
        tensors = {
            name: record.get(name)
            for name in ("input_ids", "hidden_states", "target", "last_hidden_states")
        }

        # .pt 保留源 shape，put 仍会按传输契约把 hidden/target 转为 BF16（若需要）；
        # 返回描述符必须取 store_meta，不能直接用文件内转换前的 dtype 描述实际传输字节。
        store_meta = self._mooncake_store.put(
            key=key,
            hidden_states=tensors["hidden_states"],
            input_ids=tensors["input_ids"],
            last_hidden_states=tensors["last_hidden_states"],
            target=tensors["target"],
        )

        metadata = dict(record.get("metadata") or {})
        return {
            "mooncake_key": key,
            "tensor_shapes": store_meta["shapes"],
            "tensor_dtypes": store_meta["dtypes"],
            "packed_loss_mask": record.get("packed_loss_mask"),
            "metadata": metadata,
        }

    # 以磁盘记录为准，接口中的实时输入和返回标志不参与回放；逐 ID 发布后 flush 确保读者可取数据。
    def generate(
        self,
        data_id: str | list[str],
        input_ids_ref: ray.ObjectRef | list[torch.Tensor] | None = None,
        packed_loss_mask_list: list[str] | None = None,
        formatted_prompts: list[str] | None = None,
        return_last_hidden_states: bool = False,
        return_logits: bool = True,
        multimodal_inputs: list[dict] | None = None,
    ) -> list[dict[str, Any]]:
        del (
            input_ids_ref,
            packed_loss_mask_list,
            formatted_prompts,
            return_last_hidden_states,
            return_logits,
            multimodal_inputs,
        )
        # 控制器的离线条目只带 ID/seq_len，manager 仍经普通 generate 接口调度。
        # 请求顺序如 ["b","a"]，返回也按 [b,a]；manager 用 zip 把输出与原输入配对，
        # 所以此处无需在输出再带 data_id，但绝不能自行按 manifest 顺序重排结果。
        data_ids = data_id if isinstance(data_id, list) else [data_id]
        outputs = [self._replay_one(str(item)) for item in data_ids]
        # 所有 put 后再统一等待发布完成；只有 flush 正常返回，manager 才收到可立即读取的 keys。
        # 若中途某个 ID 缺失或 put/flush 失败，前面已发布的对象不会在此回滚；整批不是事务。
        self._mooncake_store.flush()
        return outputs

    # 数据集与存储都存在才算就绪；不扫描磁盘或检测远端服务健康。
    def health_check(self, timeout: float = 5.0) -> bool:
        del timeout
        return self._dataset is not None and self._mooncake_store is not None

    # 关闭客户端并清空索引和数据集引用，释放回放占用的本地资源。
    def shutdown(self) -> None:
        if self._mooncake_store is not None:
            self._mooncake_store.close()
            self._mooncake_store = None
        self._dataset = None
        self._rows_by_id.clear()
        logger.info("OfflineReplayEngine rank %d shutdown complete", self.rank)

    # 返回记录数和数据路径，便于确认 actor 加载了哪一批离线样本。
    def get_status(self) -> dict:
        return {
            "rank": self.rank,
            "initialized": self._dataset is not None,
            "records": len(self._rows_by_id),
            "data_path": getattr(self.args, "offline_data_path", None),
        }
