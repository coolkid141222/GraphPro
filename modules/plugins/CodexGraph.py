import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.plugins.GraphProPluginModel import GraphProPluginModel
from modules.utils import EdgelistDrop
from modules.utils import scatter_add, scatter_sum
from utils.parse_args import args


init = nn.init.xavier_uniform_
logger = logging.getLogger("train_logger")


class CodexGraph(GraphProPluginModel):
    """
    A small GraphPro-style plugin modification for the thesis mainline.

    It keeps the official GraphPro pipeline and only adds:
    1. learnable fusion between structural bi-norm and relative time norm
    2. learnable layer aggregation instead of plain summation
    3. a light residual gate during finetuning
    """

    def __init__(self, dataset, pretrained_model=None, phase="pretrain"):
        super().__init__(dataset, pretrained_model, phase)
        self.edge_dropout = EdgelistDrop()

        if self.phase == "finetune":
            self.edge_mix_logit = nn.Parameter(torch.zeros(1))
            self.layer_logits = nn.Parameter(torch.zeros(args.num_layers + 1))
            self.residual_weight = nn.Parameter(init(torch.empty(args.emb_size, args.emb_size)))
            self.residual_bias = nn.Parameter(torch.zeros(1, args.emb_size))
        else:
            self.edge_mix_logit = None
            self.layer_logits = None
            self.residual_weight = None
            self.residual_bias = None

    def _agg(self, all_emb, edges, edge_norm):
        src_emb = all_emb[edges[:, 0]]
        src_emb = src_emb * edge_norm.unsqueeze(1)
        dst_emb = scatter_sum(src_emb, edges[:, 1], dim=0, dim_size=self.num_users + self.num_items)
        return dst_emb

    def _edge_binorm(self, edges):
        user_degs = scatter_add(torch.ones_like(edges[:, 0]), edges[:, 0], dim=0, dim_size=self.num_users)
        user_degs = user_degs[edges[:, 0]]
        item_degs = scatter_add(torch.ones_like(edges[:, 1]), edges[:, 1], dim=0, dim_size=self.num_items)
        item_degs = item_degs[edges[:, 1]]
        norm = torch.pow(user_degs, -0.5) * torch.pow(item_degs, -0.5)
        return norm

    def _mix_edge_signal(self, edges, edge_norm, edge_times):
        if edge_times is None:
            return edge_norm
        time_norm = self._relative_edge_time_encoding(edges, edge_times)
        if self.phase == "finetune":
            alpha = torch.sigmoid(self.edge_mix_logit)
            logger.debug(f"edge alpha: {alpha.item():.4f}")
            return edge_norm * (1.0 - alpha) + time_norm * alpha
        return edge_norm * 0.5 + time_norm * 0.5

    def _apply_residual_gate(self, all_emb):
        gated = self.emb_gate(all_emb)
        if self.phase != "finetune":
            return gated
        residual = torch.sigmoid(torch.matmul(all_emb, self.residual_weight) + self.residual_bias)
        return gated + 0.1 * torch.mul(all_emb, residual)

    def _aggregate_layers(self, layer_embs):
        if self.phase != "finetune":
            return sum(layer_embs)
        weights = torch.softmax(self.layer_logits, dim=0)
        out = 0.0
        for idx, emb in enumerate(layer_embs):
            out = out + emb * weights[idx]
        return out

    def forward(self, edges, edge_norm, edge_times):
        edge_norm = self._mix_edge_signal(edges, edge_norm, edge_times)
        all_emb = torch.cat([self.user_embedding, self.item_embedding], dim=0)
        all_emb = self._apply_residual_gate(all_emb)
        layer_embs = [all_emb]
        for _ in range(args.num_layers):
            all_emb = self._agg(layer_embs[-1], edges, edge_norm)
            layer_embs.append(all_emb)
        res_emb = self._aggregate_layers(layer_embs)
        user_res_emb, item_res_emb = res_emb.split([self.num_users, self.num_items], dim=0)
        return user_res_emb, item_res_emb

    def cal_loss(self, batch_data):
        edges, dropout_mask = self.edge_dropout(self.edges, 1 - args.edge_dropout, return_mask=True)
        edge_norm = self.edge_norm[dropout_mask]
        edge_times = self.edge_times[dropout_mask] if self.phase not in ["vanilla"] else None

        users, pos_items, neg_items = batch_data
        user_emb, item_emb = self.forward(edges, edge_norm, edge_times)
        batch_user_emb = user_emb[users]
        pos_item_emb = item_emb[pos_items]
        neg_item_emb = item_emb[neg_items]
        rec_loss = self._bpr_loss(batch_user_emb, pos_item_emb, neg_item_emb)
        reg_loss = args.weight_decay * self._reg_loss(users, pos_items, neg_items)

        loss = rec_loss + reg_loss
        loss_dict = {
            "rec_loss": rec_loss.item(),
            "reg_loss": reg_loss.item(),
        }
        return loss, loss_dict

    @torch.no_grad()
    def generate(self):
        return self.forward(self.edges, self.edge_norm, self.edge_times)

    @torch.no_grad()
    def rating(self, user_emb, item_emb):
        return torch.matmul(user_emb, item_emb.t())

    def _reg_loss(self, users, pos_items, neg_items):
        u_emb = self.user_embedding[users]
        pos_i_emb = self.item_embedding[pos_items]
        neg_i_emb = self.item_embedding[neg_items]
        reg_loss = (
            (u_emb.norm(2).pow(2) + pos_i_emb.norm(2).pow(2) + neg_i_emb.norm(2).pow(2))
            / (2 * float(len(users)))
        )
        return reg_loss
