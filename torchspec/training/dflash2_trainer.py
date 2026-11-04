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

# DFlash2 的轻量构建适配：保留 DFlash 训练循环，替换配置、网络及带 selector 损失的 wrapper。

"""DFlash2 trainer."""

from argparse import Namespace

from torchspec.models.dflash2 import DFlash2Model
from torchspec.models.draft.dflash2 import DFlash2Config, DFlash2DraftModel
from torchspec.training.dflash_trainer import DFlashTrainer


class DFlash2Trainer(DFlashTrainer):
    _draft_config_class = DFlash2Config
    _extra_loss_component_keys = ["selector_loss"]

    # 继承基础 block 训练参数，再记录 candidate selector 辅助损失的系数。
    def __init__(self, args: Namespace):
        super().__init__(args)
        self.selector_loss_alpha = getattr(args, "dflash2_selector_loss_alpha", 1.0)

    # 严格核对 block_size 和目标层数在训练参数与网络配置间一致，避免张量布局隐式错配。
    def _build_draft_model(self, config):
        if config.block_size != self.block_size:
            raise ValueError(
                "training.dflash_block_size must match dflash_config.block_size "
                f"({self.block_size} != {config.block_size})"
            )
        if config.num_target_layers != self.num_target_layers:
            raise ValueError(
                "training.dflash_num_target_layers must match the number of "
                f"dflash_config.target_layer_ids ({self.num_target_layers} != "
                f"{config.num_target_layers})"
            )
        return DFlash2DraftModel(config)

    # 将共用损失配置及 selector_loss_alpha 传给 DFlash2Model，额外指标由父类统一池化。
    def _build_training_wrapper(self, draft_model):
        return DFlash2Model(
            draft_model=draft_model,
            block_size=self.block_size,
            num_anchors=self.num_anchors,
            loss_objective=self.loss_objective,
            dpace_alpha=self.dpace_alpha,
            loss_decay_gamma=self.loss_decay_gamma,
            ce_loss_alpha=self.ce_loss_alpha,
            l1_loss_alpha=self.l1_loss_alpha,
            selector_loss_alpha=self.selector_loss_alpha,
        )
