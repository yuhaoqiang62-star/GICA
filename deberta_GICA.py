from torch import nn
from transformers.models.deberta_v2.modeling_deberta_v2 import DebertaV2PreTrainedModel, DebertaV2Model
from transformers.models.bert.modeling_bert import BertPooler
from GICA import GICA
import global_configs
from global_configs import DEVICE


class GICA_DebertaModel(DebertaV2PreTrainedModel):
    def __init__(self, config, multimodal_config):
        super().__init__(config)
        TEXT_DIM, ACOUSTIC_DIM, VISUAL_DIM = (
            global_configs.TEXT_DIM, global_configs.ACOUSTIC_DIM, global_configs.VISUAL_DIM
        )

        self.pooler = BertPooler(config)
        self.model = DebertaV2Model.from_pretrained("microsoft/deberta-v3-base").to(DEVICE)

        GICA_args = {
            'X0_dim': TEXT_DIM,
            'X1_dim': ACOUSTIC_DIM,
            'X2_dim': VISUAL_DIM,
            'B0_dim': multimodal_config.B0_dim,
            'B1_dim': multimodal_config.B1_dim,
            'inter_dim': multimodal_config.inter_dim,
            'max_sen_len': multimodal_config.max_seq_length,
            'drop_prob': multimodal_config.drop_prob,
            'p_beta': multimodal_config.p_beta,
            'p_gamma': multimodal_config.p_gamma,
            'p_lambda': multimodal_config.p_lambda,
            # 🔥 添加门控模式配置
            'gating_mode': getattr(multimodal_config, 'gating_mode', 'dual_gating'),
        }

        self.GICA = GICA(GICA_args)
        self.expand = nn.Linear(multimodal_config.B1_dim, TEXT_DIM)

        # 🔥 为声学和视觉重构特征添加投影层
        self.acoustic_proj = nn.Linear(ACOUSTIC_DIM, TEXT_DIM)
        self.visual_proj = nn.Linear(VISUAL_DIM, TEXT_DIM)

        self.LayerNorm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(multimodal_config.dropout_prob)
        self.beta_shift = multimodal_config.beta_shift

        # 🔥 添加消融实验控制参数
        self.fusion_mode = getattr(multimodal_config, 'fusion_mode', 'full')
        # fusion_mode 可选值:
        # 'b1_only': 只使用B1 (文本压缩信息)
        # 'b1_acoustic': B1 + 声学重构
        # 'b1_visual': B1 + 视觉重构
        # 'full': B1 + 声学重构 + 视觉重构 (默认)

        # 🔥 添加门控模式配置
        self.gating_mode = getattr(multimodal_config, 'gating_mode', 'dual_gating')
        # gating_mode 可选值:
        # 'no_gating': 移除所有门控
        # 'single_gating': 仅第一层使用门控
        # 'dual_gating': 两层都使用门控 (默认)

        self.init_weights()

    def forward(self, input_ids, visual, acoustic, attention_mask=None, epoch=0, max_epochs=40):
        embedding_output = self.model(input_ids, attention_mask=attention_mask)
        x = embedding_output[0]  # token-level表征

        b1, IB_total, kl_loss_0, mse_0, kl_loss_1, mse_1, intermediate_results = self.GICA(
            x, acoustic, visual, epoch, max_epochs
        )

        # 获取重构结果
        reconstructions = intermediate_results['reconstructions']

        # 扩展B1到TEXT_DIM
        h_m = self.expand(reconstructions['b1'])  # 使用纯B1特征

        # 🔥 根据消融模式选择融合策略
        if self.fusion_mode == 'b1_only':
            # 方案1: 只使用B1 (文本压缩信息)
            acoustic_vis_embedding = self.beta_shift * h_m

        elif self.fusion_mode == 'b1_acoustic':
            # 方案2: B1 + 声学重构
            acoustic_recon = reconstructions['acoustic_recon']  # [batch, seq_len, acoustic_dim]
            acoustic_proj = self.acoustic_proj(acoustic_recon)  # 投影到TEXT_DIM
            acoustic_vis_embedding = self.beta_shift * (h_m + acoustic_proj)

        elif self.fusion_mode == 'b1_visual':
            # 方案3: B1 + 视觉重构
            visual_recon = reconstructions['visual_recon']  # [batch, seq_len, visual_dim]
            visual_proj = self.visual_proj(visual_recon)  # 投影到TEXT_DIM
            acoustic_vis_embedding = self.beta_shift * (h_m + visual_proj)

        elif self.fusion_mode == 'full':
            # 方案4: B1 + 声学重构 + 视觉重构 (完整模型)
            acoustic_recon = reconstructions['acoustic_recon']
            visual_recon = reconstructions['visual_recon']
            acoustic_proj = self.acoustic_proj(acoustic_recon)
            visual_proj = self.visual_proj(visual_recon)
            acoustic_vis_embedding = self.beta_shift * (h_m + acoustic_proj + visual_proj)

        else:
            raise ValueError(f"Unknown fusion_mode: {self.fusion_mode}")

        sequence_output = self.dropout(self.LayerNorm(acoustic_vis_embedding + x))
        pooled_output = self.pooler(sequence_output)

        return pooled_output, IB_total, kl_loss_0, mse_0, kl_loss_1, mse_1


class GICA_DeBertaForSequenceClassification(DebertaV2PreTrainedModel):
    def __init__(self, config, multimodal_config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.dberta = GICA_DebertaModel(config, multimodal_config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(self, input_ids, visual, acoustic, attention_mask=None, epoch=0, max_epochs=40):
        pooled_output, IB_total, kl_loss_0, mse_0, kl_loss_1, mse_1 = self.dberta(
            input_ids, visual, acoustic, attention_mask=attention_mask, epoch=epoch, max_epochs=max_epochs
        )

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits, IB_total, kl_loss_0, mse_0, kl_loss_1, mse_1