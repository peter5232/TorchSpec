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

# 训练性能观测：PyTorch profiler 用调度窗口采集算子，内存 profiler 记录 GPU 分配历史或 CPU 原生分配。

import time
import traceback
from pathlib import Path

import torch

from torchspec.utils.logging import logger
from torchspec.utils.memory import print_memory


class TrainProfiler:
    # 按 profile_target 创建整体训练 profiler；内存历史可从初始化阶段开始，普通算子采样稍后启动。
    def __init__(self, args):
        self.args = args
        self._torch_profiler_overall = None
        self._memory_profiler_overall = None

        if args.use_pytorch_profiler and ("train_overall" in args.profile_target):
            self._torch_profiler_overall = _create_torch_profiler(args, name="train_overall")

        if args.record_memory_history and ("train_overall" in args.profile_target):
            self._memory_profiler_overall = _BaseMemoryProfiler.create(args)
            self._memory_profiler_overall.start()

    # 模型初始化完成后才 start 整体 PyTorch profiler，避免初始化淹没正式训练采样窗口。
    def on_init_end(self):
        if self._torch_profiler_overall is not None:
            self._torch_profiler_overall.start()

    # 推进整体 profiler 调度；到指定步数时停止内存记录并落盘，步号比较采用 0 起点。
    def step(self, step: int):
        if self._torch_profiler_overall is not None:
            self._torch_profiler_overall.step()

        if (
            self._memory_profiler_overall is not None
            and ((s := self.args.memory_snapshot_num_steps) is not None)
            and (step == s - 1)
        ):
            self._memory_profiler_overall.stop()

    # 把训练数据迭代器交给局部循环 profiler，保留原元素顺序和消费语义。
    def iterate_train_actor(self, iterator):
        return _profile_simple_loop(iterator, self.args, name="train_actor")


# 未启用直接 yield 原流；启用后每次训练消费完 yield 的元素，才推进 profiler.step。
def _profile_simple_loop(iterator, args, name):
    if not (args.use_pytorch_profiler and (name in args.profile_target)):
        yield from iterator
        return

    torch_profiler = _create_torch_profiler(args, name=name)
    torch_profiler.start()
    for item in iterator:
        yield item
        torch_profiler.step()


# 配置 wait/warmup/active 窗口及 rank 命名的 gzip trace，记录形状、栈、内存和 FLOPs 信息。
def _create_torch_profiler(args, name):
    return torch.profiler.profile(
        schedule=torch.profiler.schedule(
            wait=max(args.profile_step_start - 1, 0),
            warmup=1 if args.profile_step_start > 0 else 0,
            active=args.profile_step_end - args.profile_step_start,
            repeat=1,
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            args.profile_dir_name,
            worker_name=f"{name}_rank_{torch.distributed.get_rank()}",
            use_gzip=True,
        ),
        record_shapes=True,
        with_stack=True,
        profile_memory=True,
        with_flops=True,
    )


class _BaseMemoryProfiler:
    # 根据 memory_recorder 选择 torch 或 memray 实现，未知键由字典查找直接报错。
    @staticmethod
    def create(args):
        c = {
            "torch": _TorchMemoryProfiler,
            "memray": _MemrayMemoryProfiler,
        }[args.memory_recorder]
        return c(args)

    # 用时间和 rank 构造唯一快照路径，避免多个训练进程写同一文件。
    def __init__(self, args):
        self._path_dump = (
            Path(args.memory_snapshot_dir)
            / f"memory_snapshot_time{time.time()}_rank{torch.distributed.get_rank()}_{args.memory_snapshot_path}"
        )

    # 抽象启动约定：基类直接抛 NotImplementedError，由具体后端安装记录器。
    def start(self):
        raise NotImplementedError

    # 抽象停止约定：具体实现负责落盘和结束记录，基类不提供默认行为。
    def stop(self):
        raise NotImplementedError


class _TorchMemoryProfiler(_BaseMemoryProfiler):
    # 启用 CUDA 分配历史，并注册 OOM observer，使出错时也能获取分配快照和 Python 栈。
    def start(self):
        logger.info("Attach OOM dump memory history.")

        torch.cuda.memory._record_memory_history(
            max_entries=1000000,
            stacks="all",
        )

        # OOM 回调打印设备分配情况/调用栈，立即写内存快照并记录设备内存指标。
        def oom_observer(device, alloc, device_alloc, device_free):
            logger.info(
                f"Observe OOM, will dump snapshot to {self._path_dump}. ({device=} {alloc=} {device_alloc=} {device_free=}; stacktrace is as follows)"
            )
            traceback.print_stack()
            torch.cuda.memory._dump_snapshot(self._path_dump)
            print_memory("when oom")

        torch._C._cuda_attach_out_of_memory_observer(oom_observer)

    # 写最后一份 CUDA 内存快照，再关闭分配历史记录，释放持续记录的开销。
    def stop(self):
        logger.info(f"Dump memory snapshot to: {self._path_dump}")
        torch.cuda.memory._dump_snapshot(self._path_dump)
        torch.cuda.memory._record_memory_history(enabled=None)


class _MemrayMemoryProfiler(_BaseMemoryProfiler):
    # 复用快照路径并要求给定停止步数，防止 CPU 原生分配追踪无限增长。
    def __init__(self, args):
        super().__init__(args)
        assert args.memory_snapshot_num_steps is not None, (
            "In memray, must provide --memory-snapshot-num-steps"
        )

    # 按需导入 memray 并手动进入 Tracker 上下文，记录包括 native traces 的主机分配。
    def start(self):
        logger.info("Memray tracker started.")
        import memray

        self._tracker = memray.Tracker(
            file_name=self._path_dump,
            native_traces=True,
        )
        self._tracker.__enter__()

    # 退出 Tracker 上下文，结束追踪并完成快照文件写入。
    def stop(self):
        logger.info(f"Memray tracker stopped and dump snapshot to: {self._path_dump}")
        self._tracker.__exit__(None, None, None)
