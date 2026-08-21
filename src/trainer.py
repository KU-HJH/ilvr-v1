from trl import SFTTrainer
import torch
import torch.distributed as distributed
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import logging
import weakref
import deepspeed
from dataclasses import dataclass
from typing import Any, List, Mapping
from transformers import TrainerCallback

from evidence_window.config import EvidenceWindowConfig
from evidence_window.engine import (
    ConditionExecutionPlan,
    EvidenceWindowEngine,
    EvidenceWindowResult,
    ModelConditionOutput,
    build_condition_execution_plan,
    pad_evidence_layout,
)
from evidence_window.layout import InterleavedLayout, TokenIds, extract_interleaved_layout
from evidence_window.metrics import MatrixJsonlWriter, reduce_sufficient_statistics
from evidence_window.reference import build_reference_provider
from evidence_window.scores import chunked_causal_target_logps

try:
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_multimodal_rotary_pos_emb
except Exception:
    apply_multimodal_rotary_pos_emb = None

try:
    import deepspeed
    _HAS_DS = True
except ImportError:
    _HAS_DS = False


class _EMATeacher:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.teacher = copy.deepcopy(model)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, student: nn.Module):
        d = self.decay
        for p_t, p_s in zip(self.teacher.parameters(), student.parameters()):
            p_t.data.mul_(d).add_(p_s.data, alpha=(1.0 - d))


class _EMAOptimizerStepCallback(TrainerCallback):
    """Advance the trainer-owned EMA after one successful optimizer step."""

    def __init__(self, trainer: "CustomTrainerStage1") -> None:
        self._trainer_ref = weakref.ref(trainer)

    def on_optimizer_step(self, args, state, control, **kwargs):
        del args, state, control, kwargs
        trainer = self._trainer_ref()
        if trainer is None:
            return
        if bool(getattr(trainer.accelerator, "optimizer_step_was_skipped", False)):
            return
        trainer._update_ema_after_optimizer_step()


@dataclass
class _ILVRLossContext:
    loss: torch.Tensor
    outputs: object
    latent_hidden_states: torch.Tensor | None
    image_out_mask: torch.Tensor | None
    model_inputs: dict[str, Any]


class CustomTrainerStage1(SFTTrainer):
    def __init__(
        self,
        *args,
        sim_weight: float = 1.0,
        ema_tau: float = 0.999,
        coverage_p: float = 0.9,
        image_pool_k: int = 8,
        ce_weight: float = 1.0,
        helper_group_L: int = 256,
        evidence_window_config: EvidenceWindowConfig | None = None,
        latent_token_ids: Mapping[str, Any] | TokenIds | None = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.sim_weight = float(sim_weight)
        self.coverage_p = float(coverage_p)
        self.helper_group_L = int(helper_group_L)
        self.ce_weight = float(ce_weight)
        self.image_pool_k = int(image_pool_k)
        self._ema = _EMATeacher(self.model, decay=float(ema_tau))
        self._ema_update_count = 0
        if not any(
            isinstance(callback, _EMAOptimizerStepCallback)
            for callback in self.callback_handler.callbacks
        ):
            self.add_callback(_EMAOptimizerStepCallback(self))
        self.evidence_window_config = (
            EvidenceWindowConfig.disabled()
            if evidence_window_config is None
            else evidence_window_config
        )
        if not isinstance(self.evidence_window_config, EvidenceWindowConfig):
            raise TypeError("evidence_window_config must be an EvidenceWindowConfig")
        self.evidence_window_config.validate()
        self._ew_engine = None
        self._reference = None
        self._token_ids = None
        self._ew_pending_metrics = None
        self._ew_matrix_writer = None
        self._ew_matrix_observation_counter = 0
        self._ew_last_matrix_dump_step = None
        if not self.evidence_window_config.is_disabled:
            self._ew_pending_metrics = {"train": {}, "eval": {}}
            self._token_ids = self._resolve_token_ids(latent_token_ids)
            self._ew_engine = EvidenceWindowEngine(self.evidence_window_config)
            if self.evidence_window_config.residual.lambda_logit > 0:
                self._output_embedding_weight(self.model, role="policy")
                self._output_embedding_weight(self._ema.teacher, role="ema")
            self._reference = build_reference_provider(
                self.evidence_window_config.reference,
                self.model,
                self.accelerator,
            )
            self._log_evidence_window_startup()

    @staticmethod
    def _resolve_token_ids(
        configured: Mapping[str, Any] | TokenIds | None,
    ) -> TokenIds:
        if isinstance(configured, TokenIds):
            return configured
        if configured is None or not isinstance(configured, Mapping):
            raise ValueError("enabled evidence-window training requires latent_token_ids")
        required = {"start", "end", "pad", "padding", "special"}
        missing = required - set(configured)
        if missing:
            raise ValueError(
                "latent_token_ids missing required key(s): " + ", ".join(sorted(missing))
            )
        scalar_values = {}
        for source, target in (
            ("start", "latent_start"),
            ("end", "latent_end"),
            ("pad", "latent_pad"),
            ("padding", "padding"),
        ):
            value = configured[source]
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"latent_token_ids.{source} must be an integer")
            scalar_values[target] = int(value)
        special = configured["special"]
        if isinstance(special, (str, bytes)):
            raise ValueError("latent_token_ids.special must be an iterable of integers")
        try:
            special_values = frozenset(special)
        except TypeError as exc:
            raise ValueError(
                "latent_token_ids.special must be an iterable of integers"
            ) from exc
        if any(isinstance(value, bool) or not isinstance(value, int) for value in special_values):
            raise ValueError("latent_token_ids.special must contain only integers")
        return TokenIds(special=frozenset(int(value) for value in special_values), **scalar_values)

    def _log_evidence_window_startup(self) -> None:
        config = self.evidence_window_config
        logging.info(
            "Evidence-window enabled: intervention=%s, max_loo_spans_per_example=%d, "
            "reference=%s, ema_teacher=true",
            config.loo.intervention,
            config.loo.max_spans_per_example,
            config.reference.mode,
        )
        logging.info(
            "Evidence-window objective: ce_weight=%s, sim_weight=%s, beta=%s, "
            "lambda_global=%s, "
            "lambda_window=%s, lambda_ew=%s, lambda_jsd=%s, "
            "lambda_magnitude=%s, lambda_logit=%s, relation_temperature=%s, "
            "relation_huber_delta=%s, residual_metric=%s, "
            "residual_huber_delta=%s, vocab_chunk_size=%d, "
            "max_residual_tokens_per_window=%d; diagnostics: "
            "global_saturation=%s, contribution_threshold=%s, "
            "matrix_dump_interval=%d, matrix_dump_dir=%s",
            self.ce_weight,
            self.sim_weight,
            config.preference.beta,
            config.preference.lambda_global,
            config.preference.lambda_window,
            config.preference.lambda_ew,
            config.relation.lambda_jsd,
            config.relation.lambda_magnitude,
            config.residual.lambda_logit,
            config.relation.temperature,
            config.relation.huber_delta,
            config.residual.metric,
            config.residual.huber_delta,
            config.residual.vocab_chunk_size,
            config.residual.max_tokens_per_window,
            str(config.diagnostics.global_saturation).lower(),
            config.diagnostics.contribution_threshold,
            config.diagnostics.matrix_dump_interval,
            config.diagnostics.matrix_dump_dir,
        )
        if self._reference is not None:
            witness = self._reference.witness
            zero_stage = self._deepspeed_zero_stage()
            logging.info(
                "Evidence-window fixed reference: parameters=%d, parameter_bytes=%d, "
                "dtypes=%s, zero_stage=%s, ema_coexists=true",
                witness.parameter_count,
                witness.parameter_bytes,
                ",".join(witness.dtypes),
                "none" if zero_stage is None else zero_stage,
            )
            if config.reference.mode == "frozen_copy" and zero_stage == 2:
                logging.info(
                    "Evidence-window frozen reference is rank-local replicated under ZeRO-2"
                )
        nonzero_dropout = [
            f"{name}={module.p}"
            for name, module in self.model.named_modules()
            if isinstance(module, nn.Dropout) and float(module.p) > 0
        ]
        if nonzero_dropout:
            logging.warning(
                "Evidence-window condition contrasts have nonzero dropout: %s",
                ", ".join(nonzero_dropout),
            )

    def _deepspeed_zero_stage(self) -> int | None:
        state = getattr(self.accelerator, "state", None)
        plugin = getattr(state, "deepspeed_plugin", None)
        if plugin is None:
            return None
        stage = getattr(plugin, "zero_stage", None)
        return int(stage) if stage is not None else None

    def _find_latent_segments(self, input_ids, latent_start_id, latent_end_id, latent_pad_id):
        ids = input_ids[0].tolist()
        segments = []
        t = 0
        T = len(ids)
        while t < T:
            if ids[t] == latent_start_id:
                s = t
                e = s + 1
                while e < T and ids[e] != latent_end_id:
                    e += 1
                pad_pos = [i for i in range(s, e+1) if ids[i] == latent_pad_id]
                segments.append(pad_pos)
                t = e + 1
            else:
                t += 1
        return segments

    def _build_firstK_mask(self, input_ids, segments, K_list):
        B, T = input_ids.shape
        mask = torch.zeros(B, T, dtype=torch.bool, device=input_ids.device)
        assert len(segments) == len(K_list)
        for pads, K in zip(segments, K_list):
            if K > 0 and len(pads) > 0:
                take = min(K, len(pads))
                for i in pads[:take]:
                    mask[0, i] = True
        return mask

    def _top_p_top_k(self, scores: torch.Tensor, p: float, k: int):
        if scores.numel() == 0 or k <= 0:
            return []
        probs = torch.softmax(scores, dim=0)
        sorted_probs, idx = torch.sort(probs, descending=True)
        cum = torch.cumsum(sorted_probs, dim=0)
        pos = torch.nonzero(cum >= p, as_tuple=False)
        if pos.numel() > 0:
            Kp = int(pos[0].item()) + 1
        else:
            Kp = int(probs.numel())
        Kstar = min(Kp, int(k))
        return idx[:Kstar].tolist()

    def _grouped_mean(self, ei: torch.Tensor, k: int) -> torch.Tensor:
        if k <= 0:
            return ei.new_zeros((0, ei.shape[-1]))
        P = ei.shape[0]
        if P >= k and (P % k) != 0:
            newP = P - (P % k)
            if newP > 0:
                ei = ei[:newP]
            else:
                return ei.new_zeros((k, ei.shape[-1]))
        if ei.shape[0] >= k and ei.shape[0] % k == 0:
            g = ei.view(k, ei.shape[0]//k, ei.shape[-1]).mean(dim=1)
            return g
        if ei.shape[0] == 0:
            return ei.new_zeros((k, ei.shape[-1]))
        rep = math.ceil(k / ei.shape[0])
        g = ei.repeat(rep, 1)[:k]
        return g

    def _maybe_group_for_helper(self, ei: torch.Tensor, L_group: int) -> torch.Tensor:
        if L_group is None or L_group <= 0:
            return ei
        P = int(ei.shape[0])
        if P < L_group:
            return ei
        return self._grouped_mean(ei, L_group)

    def _prefix_text_mean_from_embeds(self, token_embeds: torch.Tensor, input_ids: torch.Tensor,
                                      upto_idx: int, special_ids: set) -> torch.Tensor:
        H = token_embeds.shape[-1]
        t_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        if upto_idx > 0:
            t_mask[:, :upto_idx] = 1
        for sid in special_ids:
            t_mask &= (input_ids != sid)
        feats = token_embeds[0, t_mask[0]]
        if feats.numel() == 0:
            feats = token_embeds[0, :max(1, upto_idx)]
        return feats.mean(dim=0, keepdim=True)

    @torch.no_grad()
    def _image_input_global_mean(self, tea, inputs) -> torch.Tensor:
        pv  = inputs.get("pixel_values", None)
        thw = inputs.get("image_grid_thw", None)
        if pv is None or thw is None:
            return None
        pv  = pv.to(next(tea.parameters()).device).type(tea.visual.dtype)
        thw = thw.to(next(tea.parameters()).device)
        patches = tea.visual(pv, grid_thw=thw)
        num_imgs = thw.shape[0]
        if num_imgs == 0 or patches.numel() == 0:
            return None

        s_merge = int(getattr(tea.visual, "spatial_merge_size", 2))
        thw_long = thw.to(dtype=torch.long)
        tokens_per_img = thw_long[:,0] * (thw_long[:,1]//s_merge) * (thw_long[:,2]//s_merge)
        ends = torch.cumsum(tokens_per_img, dim=0).tolist()
        starts = [0] + ends[:-1]
        outs = []
        for st, ed in zip(starts, ends):
            ei = patches[st:ed, :]
            gk = self._grouped_mean(ei, self.image_pool_k)
            outs.append(gk)
        all_groups = torch.cat(outs, dim=0)
        return all_groups.mean(dim=0, keepdim=True)

    def _get_special_ids(self):
        tok = self.tokenizer
        get_id = lambda s: tok(s, return_tensors="pt")["input_ids"][0,0].item()
        latent_pad_id   = get_id("<|latent_pad|>")
        latent_start_id = get_id("<|latent_start|>")
        latent_end_id   = get_id("<|latent_end|>")
        special_ids = {latent_pad_id, latent_start_id, latent_end_id}
        try:
            vision_start_id = get_id("<|vision_start|>")
            vision_end_id   = get_id("<|vision_end|>")
            special_ids.update({vision_start_id, vision_end_id})
        except Exception:
            pass
        img_token_id = getattr(self.model.config, "image_token_id", None)
        if img_token_id is None:
            img_token_id = 151655
        return special_ids, int(img_token_id), int(latent_start_id), int(latent_end_id), int(latent_pad_id)

    @torch.no_grad()
    def _user_side_attn_pooling(self, tea, inputs, ids, attn) -> torch.Tensor:
        device = ids.device
        B, T = ids.shape
        cand_texts = ["<|im_start|>assistant", "<|im_start|>assistant\n"]
        cand_patterns = [
            self.tokenizer(s, return_tensors="pt")["input_ids"][0].to(device) for s in cand_texts
        ]
        start_assistant = -1
        for pat in cand_patterns:
            for i in range(0, T - pat.size(0) + 1):
                if torch.all(ids[0, i:i+pat.size(0)] == pat):
                    start_assistant = i
                    break
            if start_assistant != -1:
                break
        if start_assistant <= 0:
            return None

        special_ids, image_token_id, latent_start_id, latent_end_id, latent_pad_id = self._get_special_ids()
        id_row = ids[0, :start_assistant]
        prompt_idx = [i for i in range(id_row.size(0))
                    if (int(id_row[i].item()) not in special_ids) and (int(id_row[i].item()) != image_token_id)]
        image_idx  = (id_row == image_token_id).nonzero().view(-1).tolist()
        if len(prompt_idx) == 0 or len(image_idx) == 0:
            return None

        old_flag = bool(getattr(tea.config, "output_hidden_states", False))
        tea.config.output_hidden_states = True
        try:
            out = tea(
                input_ids=ids,
                attention_mask=attn,
                pixel_values=inputs.get("pixel_values", None),
                image_grid_thw=inputs.get("image_grid_thw", None),
                output_hidden_states=True,
                return_dict=True,
            )
            hs = out.hidden_states
        finally:
            tea.config.output_hidden_states = old_flag

        if isinstance(hs, (tuple, list)):
            H_last = hs[-1]
            H_pre  = hs[-2] if len(hs) > 1 else hs[-1]
        else:
            H_last = hs
            H_pre  = hs

        try:
            layer_last = tea.model.layers[-1].self_attn
            q = layer_last.q_proj(H_pre)
            k = layer_last.k_proj(H_pre)
            num_heads = layer_last.num_heads
            dh = q.shape[-1] // num_heads
            q = q.view(1, T, num_heads, dh).transpose(1, 2)
            kv = layer_last.num_key_value_heads
            k = k.view(1, T, kv, dh).transpose(1, 2)
            if kv != num_heads:
                rep = num_heads // kv
                k = k[:, :, None, :, :].expand(1, kv, rep, T, dh).reshape(1, num_heads, T, dh)

            q_sel = q[:, :, prompt_idx, :]
            k_sel = k[:, :, image_idx, :]
            logits = torch.einsum("bhpd,bhqd->bhpq", q_sel, k_sel) / (dh ** 0.5)
            probs_per_prompt = torch.softmax(logits, dim=-1)
            w = probs_per_prompt.mean(dim=2).mean(dim=1).squeeze(0)
            w = w / (w.sum() + 1e-9)

            V_img = H_last[0, :start_assistant, :][image_idx, :]
            V_prm = H_last[0, :start_assistant, :][prompt_idx, :]
            r_img = (w.unsqueeze(0) @ V_img).squeeze(0)
            r_txt = V_prm.mean(dim=0)
            u = torch.stack([r_img, r_txt], dim=0).mean(dim=0, keepdim=True)
            return u
        except Exception:
            H_sub = H_last[0, :start_assistant, :]
            V_img = H_sub[image_idx, :]
            V_prm = H_sub[prompt_idx, :]
            sim = (V_prm @ V_img.t()) / (V_img.shape[-1] ** 0.5)
            w = torch.softmax(sim, dim=-1).mean(dim=0)
            w = w / (w.sum() + 1e-9)
            r_img = (w.unsqueeze(0) @ V_img).squeeze(0)
            r_txt = V_prm.mean(dim=0)
            u = torch.stack([r_img, r_txt], dim=0).mean(dim=0, keepdim=True)
            return u

    def _extract_assistant_text_spans(self, ids: torch.Tensor,
                                      latent_starts: List[int], latent_ends: List[int],
                                      assistant_start: int, special_ids: set) -> List[List[int]]:
        spans = []
        prev = assistant_start
        for s, e in zip(latent_starts, latent_ends):
            span_idx = []
            for t in range(prev, s):
                tid = int(ids[t].item())
                if (tid not in special_ids):
                    span_idx.append(t)
            spans.append(span_idx)
            prev = e + 1
        return spans

    @torch.no_grad()
    def _teacher_build_latents(self, inputs, k, p):
        device = self.model.device
        ids  = inputs["input_ids"].to(device)
        attn = inputs["attention_mask"].to(device)
        tea  = self._ema.teacher.to(device).eval()

        special_ids, image_token_id, latent_start_id, latent_end_id, latent_pad_id = self._get_special_ids()

        seg_pad_indices = self._find_latent_segments(ids, latent_start_id, latent_end_id, latent_pad_id)
        if len(seg_pad_indices) == 0:
            return None, torch.zeros_like(ids, dtype=torch.bool)

        idlist = ids[0].tolist()
        starts = []
        ends   = []
        t = 0; T = len(idlist)
        while t < T:
            if idlist[t] == latent_start_id:
                s = t
                e = s + 1
                while e < T and idlist[e] != latent_end_id:
                    e += 1
                starts.append(s); ends.append(e)
                t = e + 1
            else:
                t += 1

        pat = self.tokenizer("<|im_start|>assistant", return_tensors="pt")["input_ids"][0].to(device)
        start_assistant = -1
        for i in range(0, ids.size(1) - pat.size(0) + 1):
            if torch.all(ids[0, i:i+pat.size(0)] == pat):
                start_assistant = i; break
        if start_assistant <= 0:
            return None, torch.zeros_like(ids, dtype=torch.bool)

        u = self._user_side_attn_pooling(tea, inputs, ids, attn)
        if u is None:
            token_embeds = tea.get_input_embeddings()(ids)
            u_text = self._prefix_text_mean_from_embeds(token_embeds, ids, start_assistant, special_ids)
            u_img  = self._image_input_global_mean(tea, inputs)
            parts = [u_text] + ([u_img] if u_img is not None else [])
            u = torch.stack([x.squeeze(0) for x in parts]).mean(dim=0, keepdim=True)

        pv  = inputs.get("pixel_values_latent", None)
        thw = inputs.get("image_grid_thw_latent", None)
        if pv is None or thw is None:
            return None, torch.zeros_like(ids, dtype=torch.bool)
        pv  = pv.to(device).to(tea.visual.dtype)
        thw = thw.to(device)
        patch_all = tea.visual(pv, grid_thw=thw)
        num_imgs  = int(thw.shape[0])

        s_merge = int(getattr(tea.visual, "spatial_merge_size", 2))
        thw_long = thw.to(dtype=torch.long)
        tokens_per_img = (thw_long[:,0] * (thw_long[:,1]//s_merge) * (thw_long[:,2]//s_merge))
        ends_img = torch.cumsum(tokens_per_img, dim=0).tolist()
        starts_img = [0] + ends_img[:-1]
        slices_per_img = [(int(st), int(ed)) for st, ed in zip(starts_img, ends_img)]
        assert len(slices_per_img) == num_imgs

        text_spans = self._extract_assistant_text_spans(ids[0], starts, ends, start_assistant, special_ids)
        L_group = getattr(self, "helper_group_L", None)
        if L_group is None:
            L_group = int(self.image_pool_k)

        latents_list = []
        Kstars = []
        prev_sel_mean = None

        for seg_idx, pad_pos in enumerate(seg_pad_indices):
            if len(pad_pos) == 0:
                Kstars.append(0); continue

            text_idx = text_spans[seg_idx] if seg_idx < len(text_spans) else []
            old_flag = bool(getattr(tea.config, "output_hidden_states", False))
            tea.config.output_hidden_states = True
            try:
                out_assist = tea(
                    input_ids=ids,
                    attention_mask=attn,
                    pixel_values=inputs.get("pixel_values", None),
                    image_grid_thw=inputs.get("image_grid_thw", None),
                    output_hidden_states=True,
                    return_dict=True,
                )
            finally:
                tea.config.output_hidden_states = old_flag

            hs = out_assist.hidden_states
            x = hs[-1] if isinstance(hs, (tuple, list)) else hs
            H_last_2d = x[0] if x.dim() == 3 else (x if x.dim() == 2 else tea.get_input_embeddings()(ids)[0])

            q_parts = [u]

            if len(text_idx) > 0:
                idx_tensor = torch.tensor(text_idx, device=H_last_2d.device, dtype=torch.long)
                q_parts.append(H_last_2d.index_select(0, idx_tensor).mean(dim=0, keepdim=True))

            if prev_sel_mean is not None:
                q_parts.append(prev_sel_mean)

            q_t = torch.stack([x.squeeze(0) for x in q_parts], dim=0).mean(dim=0, keepdim=True)

            assert seg_idx < num_imgs, "latent 段数量与 helper images 数量不一致"
            st_img, ed_img = slices_per_img[seg_idx]
            ei = patch_all[st_img:ed_img, :]
            cand = self._maybe_group_for_helper(ei, L_group)

            sim = F.cosine_similarity(cand, q_t.expand_as(cand), dim=-1)
            top_idx = self._top_p_top_k(sim, p=self.coverage_p, k=k)
            Kstar = len(top_idx)
            Kstars.append(Kstar)

            if Kstar > 0:
                chosen = cand[top_idx[:Kstar], :]
                latents_list.append(chosen)
                prev_sel_mean = chosen.mean(dim=0, keepdim=True)

        if len(latents_list) == 0:
            return None, torch.zeros_like(ids, dtype=torch.bool)
        latents = torch.cat(latents_list, dim=0).unsqueeze(0)

        firstK_mask = self._build_firstK_mask(ids, seg_pad_indices, Kstars)
        return latents, firstK_mask

    def _compute_ilvr_loss_context(self, model, inputs, num_items_in_batch=None):
        k = getattr(self.model.config, "latent_size", 8)
        teacher_latents, firstK_mask = self._teacher_build_latents(inputs, k=k, p=self.coverage_p)
        
        with torch.no_grad():
            pv_lat = inputs.get("pixel_values_latent", None)
            thw_lat = inputs.get("image_grid_thw_latent", None)
            if pv_lat is not None and thw_lat is not None:
                _ = self.model.visual(
                    pv_lat.to(self.model.device).to(self.model.visual.dtype),
                    grid_thw=thw_lat.to(self.model.device)
                )
        
        if teacher_latents is None:
            ce_loss, outputs = super().compute_loss(
                model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
            )
            return _ILVRLossContext(ce_loss, outputs, None, None, dict(inputs))

        if teacher_latents.dim() == 2:
            teacher_latents = teacher_latents.unsqueeze(0)
        S = int(firstK_mask.sum().item())
        if teacher_latents.shape[1] != S:
            ce_loss, outputs = super().compute_loss(
                model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
            )
            return _ILVRLossContext(ce_loss, outputs, None, None, dict(inputs))

        mod_inputs = dict(inputs)
        mod_inputs.pop("pixel_values_latent", None)
        mod_inputs["latent_hidden_states"] = teacher_latents.to(self.model.device).to(self.model.dtype)
        mod_inputs["image_out_mask"] = firstK_mask

        ce_loss, outputs = super().compute_loss(
            model, mod_inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )
        if self.sim_weight == 0.0:
            return _ILVRLossContext(
                ce_loss,
                outputs,
                mod_inputs["latent_hidden_states"],
                firstK_mask,
                mod_inputs,
            )

        pred_h = outputs.hidden_states
        inp_h  = outputs.inputs_embeds
        B, T, H = pred_h.shape
        if T <= 1:
            return _ILVRLossContext(
                ce_loss,
                outputs,
                mod_inputs["latent_hidden_states"],
                firstK_mask,
                mod_inputs,
            )

        mask = mod_inputs["image_out_mask"][:, -(T - 1):].to(pred_h.device).bool()
        if not mask.any():
            return _ILVRLossContext(
                ce_loss,
                outputs,
                mod_inputs["latent_hidden_states"],
                firstK_mask,
                mod_inputs,
            )

        pred = pred_h[..., :-1, :][mask].contiguous().float()
        gt   = inp_h[...,  1:, :][mask].contiguous().detach().float()
        gt = gt + 0.01 * torch.randn_like(gt)

        cos = F.cosine_similarity(gt, pred, dim=-1).mean()
        sim_loss = 1.0 - cos
        loss = self.ce_weight * ce_loss + self.sim_weight * sim_loss
        return _ILVRLossContext(
            loss,
            outputs,
            mod_inputs["latent_hidden_states"],
            firstK_mask,
            mod_inputs,
        )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        context = self._compute_ilvr_loss_context(model, inputs, num_items_in_batch)
        if self._ew_engine is None:
            return (context.loss, context.outputs) if return_outputs else context.loss
        batch_size = int(context.model_inputs["input_ids"].shape[0])
        batch_count = context.loss.new_tensor(batch_size).detach()
        split = "train" if model.training else "eval"
        self._accumulate_evidence_window_metrics(
            {
                "loss/ilvr": (
                    context.loss.detach()
                    * batch_count.to(dtype=context.loss.dtype),
                    batch_count,
                )
            },
            ordinal_families=(),
            split=split,
        )
        auxiliary = self._compute_evidence_window(model, context)
        metric_sums = dict(auxiliary.metric_sums)
        ordinal_families = []
        config = self.evidence_window_config
        if (
            config.preference.lambda_global > 0
            or config.preference.lambda_window > 0
            or config.diagnostics.global_saturation
        ):
            ordinal_families.append("preference/window_margin")
        if config.preference.lambda_ew > 0:
            ordinal_families.append("preference/ew_margin")
        self._accumulate_evidence_window_metrics(
            metric_sums,
            ordinal_families=ordinal_families,
            split=split,
        )
        self._maybe_dump_evidence_window_matrices(auxiliary)
        weighted_auxiliary = self._compose_globally_weighted_auxiliary_loss(
            context.loss,
            auxiliary,
        )
        loss = context.loss + weighted_auxiliary
        return (loss, context.outputs) if return_outputs else loss

    def _active_auxiliary_objectives(self) -> tuple[tuple[str, float], ...]:
        config = self.evidence_window_config
        candidates = (
            ("global_preference", config.preference.lambda_global),
            ("window_preference", config.preference.lambda_window),
            ("ew_dpo", config.preference.lambda_ew),
            ("relation_jsd", config.relation.lambda_jsd),
            ("relation_magnitude", config.relation.lambda_magnitude),
            ("logit_residual", config.residual.lambda_logit),
        )
        return tuple((key, coefficient) for key, coefficient in candidates if coefficient > 0)

    def _compose_globally_weighted_auxiliary_loss(
        self,
        zero_source: torch.Tensor,
        auxiliary: EvidenceWindowResult | None,
    ) -> torch.Tensor:
        graph_zero = zero_source * 0.0
        total = graph_zero
        world_size = (
            distributed.get_world_size()
            if distributed.is_available() and distributed.is_initialized()
            else 1
        )
        for key, coefficient in self._active_auxiliary_objectives():
            local_loss = graph_zero
            local_count = torch.zeros(
                (),
                dtype=torch.float64,
                device=zero_source.device,
            )
            if auxiliary is not None:
                local_loss = auxiliary.losses.get(key, graph_zero)
                statistics = auxiliary.metric_sums.get(f"loss/{key}")
                if statistics is not None:
                    local_count = statistics[1].detach().to(
                        device=zero_source.device,
                        dtype=torch.float64,
                    )
            global_count = local_count.clone()
            if distributed.is_available() and distributed.is_initialized():
                distributed.all_reduce(global_count, op=distributed.ReduceOp.SUM)
            if global_count.item() <= 0:
                total = total + local_loss * 0.0
                continue
            scale = (world_size * local_count / global_count).to(
                dtype=local_loss.dtype
            )
            total = total + coefficient * scale * local_loss
        return total

    def _accumulate_evidence_window_metrics(
        self,
        metric_sums: Mapping[str, tuple[torch.Tensor, torch.Tensor]],
        *,
        ordinal_families: tuple[str, ...] | list[str],
        split: str = "train",
    ) -> None:
        if not metric_sums:
            return
        if self._ew_pending_metrics is None:
            self._ew_pending_metrics = {"train": {}, "eval": {}}
        if split not in {"train", "eval"}:
            raise ValueError("metric split must be 'train' or 'eval'")
        del ordinal_families

        pending = self._ew_pending_metrics[split]
        for key, (numerator, count) in metric_sums.items():
            local_numerator = numerator.detach().to(dtype=torch.float64)
            local_count = count.detach().to(
                device=local_numerator.device,
                dtype=torch.float64,
            )
            previous = pending.get(key)
            if previous is None:
                pending[key] = (local_numerator, local_count)
            else:
                pending[key] = (
                    previous[0] + local_numerator,
                    previous[1] + local_count,
                )

    def _reduce_evidence_window_metrics(self, split: str) -> dict[str, float]:
        if split not in {"train", "eval"}:
            raise ValueError("metric split must be 'train' or 'eval'")
        if self._ew_pending_metrics is None:
            return {}

        synchronized = dict(self._ew_pending_metrics[split])
        if synchronized:
            device = next(iter(synchronized.values()))[0].device
        else:
            device = self._module_device(self.model)
        has_pending = torch.tensor(
            1.0 if synchronized else 0.0,
            dtype=torch.float64,
            device=device,
        )
        if distributed.is_available() and distributed.is_initialized():
            distributed.all_reduce(has_pending, op=distributed.ReduceOp.SUM)
        if has_pending.item() == 0:
            return {}

        zero = torch.zeros((), dtype=torch.float64, device=device)
        for key in self._expected_evidence_metric_keys():
            synchronized.setdefault(key, (zero, zero))

        for family in self._ordinal_metric_families():
            prefix = family + "/"
            local_ordinals = [
                int(key[len(prefix):])
                for key in synchronized
                if key.startswith(prefix) and key[len(prefix):].isdigit()
            ]
            max_ordinal = torch.tensor(
                max(local_ordinals, default=-1),
                dtype=torch.long,
                device=device,
            )
            if distributed.is_available() and distributed.is_initialized():
                distributed.all_reduce(max_ordinal, op=distributed.ReduceOp.MAX)
            for ordinal in range(int(max_ordinal.item()) + 1):
                synchronized.setdefault(f"{family}/{ordinal}", (zero, zero))

        reduced_metrics = {}
        for key in sorted(synchronized):
            numerator, count = synchronized[key]
            reduced = reduce_sufficient_statistics(numerator, count)
            reduced_count = float(reduced.count.detach().cpu().item())
            if reduced_count <= 0:
                continue
            reduced_numerator = float(reduced.numerator.detach().cpu().item())
            reduced_metrics[key] = (
                reduced_numerator
                if key.endswith("/entropy_valid_count")
                else reduced_numerator / reduced_count
            )
        return reduced_metrics

    def _ordinal_metric_families(self) -> tuple[str, ...]:
        config = self.evidence_window_config
        families = []
        if (
            config.preference.lambda_global > 0
            or config.preference.lambda_window > 0
            or config.diagnostics.global_saturation
        ):
            families.append("preference/window_margin")
        if config.preference.lambda_ew > 0:
            families.append("preference/ew_margin")
        return tuple(families)

    def _expected_evidence_metric_keys(self) -> set[str]:
        config = self.evidence_window_config
        keys = {"loss/ilvr"}
        need_all_null = (
            config.preference.lambda_global > 0
            or config.preference.lambda_window > 0
            or config.diagnostics.global_saturation
        )
        need_relation = (
            config.relation.lambda_jsd > 0
            or config.relation.lambda_magnitude > 0
            or config.diagnostics.matrix_dump_interval > 0
        )
        need_loo = (
            config.preference.lambda_ew > 0
            or need_relation
            or config.residual.lambda_logit > 0
        )
        if config.preference.lambda_global > 0:
            keys.add("loss/global_preference")
        if config.preference.lambda_window > 0:
            keys.add("loss/window_preference")
        if need_all_null:
            keys.update(
                {
                    "preference/global_margin",
                    "preference/global_outer_weight",
                    "preference/window_margin_mean",
                }
            )
        if config.preference.lambda_ew > 0:
            keys.update({"loss/ew_dpo", "preference/ew_margin_mean"})
        if config.relation.lambda_jsd > 0:
            keys.add("loss/relation_jsd")
        if config.relation.lambda_magnitude > 0:
            keys.add("loss/relation_magnitude")
        if need_relation:
            for role in ("student", "teacher"):
                keys.update(
                    {
                        f"relation/{role}/top1_ratio",
                        f"relation/{role}/normalized_entropy",
                        f"relation/{role}/entropy_valid_count",
                        f"relation/{role}/above_threshold_count",
                    }
                )
        if config.residual.lambda_logit > 0:
            keys.update({"loss/logit_residual", "residual/selected_token_count"})
        if need_loo:
            keys.add("loo/selected_span_count")
        return keys

    @staticmethod
    def _is_ordinal_metric(key: str) -> bool:
        return any(
            key.startswith(prefix)
            and key[len(prefix):].isdigit()
            for prefix in (
                "preference/window_margin/",
                "preference/ew_margin/",
            )
        )

    def _maybe_dump_evidence_window_matrices(
        self,
        auxiliary: EvidenceWindowResult,
    ) -> None:
        diagnostics = self.evidence_window_config.diagnostics
        interval = diagnostics.matrix_dump_interval
        if interval <= 0:
            return
        state = getattr(self, "state", None)
        step = getattr(state, "global_step", None)
        if isinstance(step, bool) or not isinstance(step, int):
            return
        if step <= 0 or step % interval != 0 or step == self._ew_last_matrix_dump_step:
            return

        relation = auxiliary.relations.get("student")
        pair_mask = auxiliary.relation_pair_mask
        if relation is None:
            relation = auxiliary.ew_margins
            pair_mask = auxiliary.preference_pair_mask
        if relation is None:
            return

        if self._ew_matrix_writer is None:
            rank = getattr(self.accelerator, "process_index", None)
            if rank is None:
                rank = (
                    distributed.get_rank()
                    if distributed.is_available() and distributed.is_initialized()
                    else 0
                )
            self._ew_matrix_writer = MatrixJsonlWriter(
                diagnostics.matrix_dump_dir,
                rank=int(rank),
            )
        self._ew_matrix_writer.write(
            step,
            self._ew_matrix_observation_counter,
            relation.detach(),
            pair_mask.detach(),
        )
        self._ew_matrix_observation_counter += 1
        self._ew_last_matrix_dump_step = step

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        pending = self._ew_pending_metrics
        if pending is None:
            super().log(logs, start_time)
            return

        is_evaluation = any(key.startswith("eval_") for key in logs)
        split = "eval" if is_evaluation else "train"
        reduced = self._reduce_evidence_window_metrics(split)
        combined = dict(logs)
        for key, value in reduced.items():
            combined[f"eval_{key}" if is_evaluation else key] = value
        super().log(combined, start_time)
        pending[split].clear()

    def _compute_evidence_window(self, model, context: _ILVRLossContext):
        if self._ew_engine is None or self._token_ids is None:
            raise RuntimeError("evidence-window runtime is not initialized")
        required = {"input_ids", "attention_mask", "labels"}
        missing = required - set(context.model_inputs)
        if missing:
            raise ValueError(
                "enabled evidence-window loss requires model input(s): "
                + ", ".join(sorted(missing))
            )
        input_ids = context.model_inputs["input_ids"]
        active_mask = context.image_out_mask
        if active_mask is None:
            active_mask = torch.zeros_like(input_ids, dtype=torch.bool)

        layout = extract_interleaved_layout(
            input_ids,
            context.model_inputs["attention_mask"],
            context.model_inputs["labels"],
            active_mask,
            self._token_ids,
        )
        layout, execution_plan = self._synchronize_evidence_window_plan(layout)
        labels = context.model_inputs["labels"]
        latent_values = context.latent_hidden_states
        if latent_values is None:
            hidden_states = self._final_hidden_states(context.outputs)
            if hidden_states is not None:
                hidden_size = int(hidden_states.shape[-1])
                latent_dtype = hidden_states.dtype
            else:
                input_embeddings = model.get_input_embeddings().weight
                hidden_size = int(input_embeddings.shape[-1])
                latent_dtype = input_embeddings.dtype
            latent_values = torch.empty(
                (input_ids.shape[0], 0, hidden_size),
                dtype=latent_dtype,
                device=input_ids.device,
            )
        policy_full = ModelConditionOutput(
            token_logps=self._evidence_token_logps(
                context.outputs.logits,
                labels.to(context.outputs.logits.device),
            ),
            hidden_states=self._final_hidden_states(context.outputs),
        )

        def forward(role, condition, latent_values, need_hidden):
            del condition
            condition_inputs = self._condition_inputs(
                context,
                latent_values,
                active_mask,
            )
            return self._forward_evidence_condition(
                role,
                model,
                condition_inputs,
                labels,
                need_hidden,
            )

        def lm_head_weight(role):
            role_model = model if role == "policy" else self._ema.teacher
            return self._output_embedding_weight(role_model, role=role)

        return self._ew_engine.compute(
            layout,
            latent_values,
            policy_full,
            forward,
            lm_head_weight=lm_head_weight,
            execution_plan=execution_plan,
        )

    def _synchronize_evidence_window_plan(
        self,
        layout: InterleavedLayout,
    ) -> tuple[InterleavedLayout, ConditionExecutionPlan]:
        """Agree on padded evidence ordinals before any auxiliary forward."""
        if self._ew_engine is None:
            raise RuntimeError("evidence-window runtime is not initialized")
        device = layout.evidence_token_mask.device
        global_count = torch.tensor(
            layout.evidence_token_mask.shape[1],
            dtype=torch.long,
            device=device,
        )
        is_distributed = distributed.is_available() and distributed.is_initialized()
        if is_distributed:
            distributed.all_reduce(global_count, op=distributed.ReduceOp.MAX)
        padded = pad_evidence_layout(layout, int(global_count.item()))
        local_indices = self._ew_engine.selected_evidence_indices(padded)
        participation = torch.zeros(
            int(global_count.item()),
            dtype=torch.long,
            device=device,
        )
        if local_indices:
            participation[list(local_indices)] = 1
        if is_distributed and participation.numel():
            distributed.all_reduce(participation, op=distributed.ReduceOp.MAX)
        global_indices = tuple(
            participation.nonzero(as_tuple=False).flatten().tolist()
        )
        return padded, build_condition_execution_plan(
            self.evidence_window_config,
            global_indices,
        )

    @staticmethod
    def _condition_inputs(
        context: _ILVRLossContext,
        latent_values: torch.Tensor,
        image_out_mask: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        values = dict(context.model_inputs)
        values.pop("labels", None)
        values.pop("pixel_values_latent", None)
        values.pop("image_grid_thw_latent", None)
        values["latent_hidden_states"] = latent_values
        values["image_out_mask"] = (
            context.image_out_mask if image_out_mask is None else image_out_mask
        )
        values["return_dict"] = True
        return values

    def _forward_evidence_condition(
        self,
        role: str,
        policy_model: nn.Module,
        inputs: Mapping[str, Any],
        labels: torch.Tensor,
        need_hidden: bool,
    ) -> ModelConditionOutput:
        if role == "policy":
            role_model = policy_model
            role_inputs = dict(inputs)
            role_labels = labels
            call = lambda: role_model(**role_inputs)
        elif role == "reference":
            if self._reference is None:
                raise RuntimeError("reference condition requested without a fixed reference")
            role_inputs = dict(inputs)
            role_labels = labels
            call = lambda: self._reference.forward(policy_model, role_inputs)
        elif role == "ema":
            role_model = self._ema.teacher
            device = self._module_device(role_model)
            role_inputs = self._move_tensors(inputs, device)
            role_labels = labels.to(device)
            call = lambda: role_model(**role_inputs)
        else:
            raise ValueError(f"unsupported evidence-window role: {role}")

        role_inputs["output_hidden_states"] = need_hidden
        if role == "policy":
            outputs = call()
        else:
            with torch.no_grad():
                outputs = call()
        token_logps = self._evidence_token_logps(
            outputs.logits,
            role_labels.to(outputs.logits.device),
        )
        hidden_states = self._final_hidden_states(outputs) if need_hidden else None
        return ModelConditionOutput(token_logps, hidden_states)

    @staticmethod
    def _evidence_token_logps(
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        target_logps = chunked_causal_target_logps(logits, labels)
        if logits.shape[1] == 0:
            return target_logps
        return torch.cat(
            (target_logps.new_zeros((logits.shape[0], 1)), target_logps),
            dim=1,
        )

    @staticmethod
    def _final_hidden_states(outputs: object) -> torch.Tensor | None:
        hidden_states = getattr(outputs, "hidden_states", None)
        if isinstance(hidden_states, (tuple, list)):
            return hidden_states[-1] if hidden_states else None
        return hidden_states

    def _output_embedding_weight(self, model: nn.Module, *, role: str) -> torch.Tensor:
        unwrapped = self.accelerator.unwrap_model(model)
        get_head = getattr(unwrapped, "get_output_embeddings", None)
        if not callable(get_head):
            raise ValueError(f"{role} model must expose get_output_embeddings()")
        output_head = get_head()
        weight = getattr(output_head, "weight", None)
        if not torch.is_tensor(weight):
            raise ValueError(f"{role} output embedding must expose a weight tensor")
        if getattr(output_head, "bias", None) is not None:
            raise ValueError(
                f"residual KD requires a bias-free {role} output embedding"
            )
        return weight

    @staticmethod
    def _module_device(module: nn.Module) -> torch.device:
        try:
            return next(module.parameters()).device
        except StopIteration as exc:
            raise ValueError("condition model must own at least one parameter") from exc

    @classmethod
    def _move_tensors(cls, value: Any, device: torch.device) -> Any:
        if torch.is_tensor(value):
            return value.to(device)
        if isinstance(value, dict):
            return {key: cls._move_tensors(item, device) for key, item in value.items()}
        if isinstance(value, tuple):
            return tuple(cls._move_tensors(item, device) for item in value)
        if isinstance(value, list):
            return [cls._move_tensors(item, device) for item in value]
        return value

    @property
    def ema_update_count(self) -> int:
        return self._ema_update_count

    @torch.no_grad()
    def _update_ema_after_optimizer_step(self) -> None:
        d = float(self._ema.decay)
        if _HAS_DS and any(hasattr(p, "ds_id") for p in self.model.parameters()):
            for p_t, p_s in zip(self._ema.teacher.parameters(), self.model.parameters()):
                with deepspeed.zero.GatheredParameters(p_s, modifier_rank=0):
                    if p_s.data.numel() == 0:
                        continue
                    p_data = p_s.data
                    if p_data.device != p_t.data.device:
                        p_data = p_data.to(p_t.data.device)
                    p_t.data.mul_(d).add_(p_data, alpha=(1.0 - d))
        else:
            self._ema.update(self.model)
        self._ema_update_count += 1
