# Code referred from:
# https://github.com/TideDancer/ctc-align/
from typing import Dict, List, Optional, Tuple, Union
from collections import Counter
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from transformers import Wav2Vec2ForCTC, Wav2Vec2Model
from transformers.modeling_outputs import CausalLMOutput
from torch import nn

from utils.entropy_loss import ctc_entropy_cost

class Wav2Vec2ForECTC(Wav2Vec2ForCTC):
    def __init__(self, config, entropy_beta=0.1):
        super().__init__(config)
        self.prior_list = [] # prior list = [ tensor of counts, size = vocab_size ], e.g. can be a uniform prior, a tensor whose elements are all 1
        self.prior = torch.ones(config.vocab_size, requires_grad=False)
        self.entropy_beta = entropy_beta

        self.wav2vec2 = Wav2Vec2Model(config)
        self.dropout = nn.Dropout(config.final_dropout)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)

        self.init_weights()

    def freeze_feature_extractor(self):
        """
        Calling this function will disable the gradient computation for the feature extractor so that its parameter
        will not be updated during training.
        """
        self.wav2vec2.feature_extractor._freeze_parameters()

    def forward(
        self,
        input_values,
        attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
    ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)

        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:

            if labels.max() >= self.config.vocab_size:
                raise ValueError(f"Label values must be <= vocab_size: {self.config.vocab_size}")

            # retrieve loss input_lengths from attention_mask
            attention_mask = (
                attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
            )
            input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)

            # assuming that padded tokens are filled with -100
            # when not being attended to
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)

            # ctc_loss doesn't support fp16
            log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)

            with torch.backends.cudnn.flags(enabled=False):
                H, cost = ctc_entropy_cost(log_probs, flattened_targets, input_lengths, target_lengths, sumed=False, blank=self.config.pad_token_id)
                H, cost = torch.mean(H), torch.mean(cost)
                # We add 200 to avoid loss going negative
                loss = cost - self.entropy_beta*H + 200.0   
                #print(f"token {self.config.pad_token_id}, beta {self.entropy_beta}, ctc_loss {cost}, H {H}, total loss {loss}")

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )
    
    
# class Wav2Vec2ForCTC_KL(Wav2Vec2ForCTC):
#     def __init__(self, config, ctc_weight=0.3, label_smoothing=0.1):
#         super().__init__(config)
    
#         self.ctc_weight = ctc_weight
#         self.label_smoothing = label_smoothing
#         self.wav2vec2 = Wav2Vec2Model(config)
#         self.dropout = nn.Dropout(config.final_dropout)

#         if config.vocab_size is None:
#             raise ValueError(
#                 f"You are trying to instantiate {self.__class__} with a configuration that "
#                 "does not define the vocabulary size of the language model head. Please "
#                 "instantiate the model as follows: `Wav2Vec2ForCTC.from_pretrained(..., vocab_size=vocab_size)`. "
#                 "or define `vocab_size` of your model's configuration."
#             )
#         output_hidden_size = (
#             config.output_hidden_size if hasattr(config, "add_adapter") and config.add_adapter else config.hidden_size
#         )
#         self.lm_head = nn.Linear(output_hidden_size, config.vocab_size)

#         # Initialize weights and apply final processing
#         self.post_init()

#     def freeze_feature_extractor(self):
#         """
#         Calling this function will disable the gradient computation for the feature encoder so that its parameter will
#         not be updated during training.
#         """
#         warnings.warn(
#             "The method `freeze_feature_extractor` is deprecated and will be removed in Transformers v5."
#             "Please use the equivalent `freeze_feature_encoder` method instead.",
#             FutureWarning,
#         )
#         self.freeze_feature_encoder()

#     def freeze_feature_encoder(self):
#         """
#         Calling this function will disable the gradient computation for the feature encoder so that its parameter will
#         not be updated during training.
#         """
#         self.wav2vec2.feature_extractor._freeze_parameters()

#     @add_start_docstrings_to_model_forward(WAV_2_VEC_2_INPUTS_DOCSTRING)
#     @add_code_sample_docstrings(
#         processor_class=_PROCESSOR_FOR_DOC,
#         checkpoint=_CHECKPOINT_FOR_DOC,
#         output_type=CausalLMOutput,
#         config_class=_CONFIG_FOR_DOC,
#         expected_output=_CTC_EXPECTED_OUTPUT,
#         expected_loss=_CTC_EXPECTED_LOSS,
#     )
#     def forward(
#         self,
#         input_values: Optional[torch.Tensor],
#         attention_mask: Optional[torch.Tensor] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#         labels: Optional[torch.Tensor] = None,
#     ) -> Union[Tuple, CausalLMOutput]:
#         r"""
#         labels (`torch.LongTensor` of shape `(batch_size, target_length)`, *optional*):
#             Labels for connectionist temporal classification. Note that `target_length` has to be smaller or equal to
#             the sequence length of the output logits. Indices are selected in `[-100, 0, ..., config.vocab_size - 1]`.
#             All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ...,
#             config.vocab_size - 1]`.
#         """

#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         outputs = self.wav2vec2(
#             input_values,
#             attention_mask=attention_mask,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )

#         hidden_states = outputs[0]
#         hidden_states = self.dropout(hidden_states)

#         logits = self.lm_head(hidden_states)

#         loss = None
#         if labels is not None:

#             if labels.max() >= self.config.vocab_size:
#                 raise ValueError(f"Label values must be <= vocab_size: {self.config.vocab_size}")

#             # retrieve loss input_lengths from attention_mask
#             attention_mask = (
#                 attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
#             )
#             input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)

#             # assuming that padded tokens are filled with -100
#             # when not being attended to
#             labels_mask = labels >= 0
#             target_lengths = labels_mask.sum(-1)
#             flattened_targets = labels.masked_select(labels_mask)

#             # ctc_loss doesn't support fp16
#             log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)
            

#             with torch.backends.cudnn.flags(enabled=False):                
                
#                 ctc_loss = nn.functional.ctc_loss(log_probs,  # p_ctc
#                                                   flattened_targets, # targets/tokens
#                                                   input_lengths,  # wav_lens
#                                                   target_lengths, # token_lens
#                                                   blank=self.config.pad_token_id, 
#                                                   reduction=self.config.ctc_loss_reduction, 
#                                                   zero_infinity=self.config.ctc_zero_infinity,
#                                                  )
                
#                 #https://github.com/speechbrain/speechbrain/blob/develop/recipes/AISHELL-1/ASR/transformer/train.py
                
#                 kl_loss = kldiv_loss(log_probs, flattened_targets, label_smoothing=0.1)
                
#                 loss = (self.ctc_weight*ctc_loss + (1 - self.ctc_weight)*kl_loss
#         )

#         if not return_dict:
#             output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
#             return ((loss,) + output) if loss is not None else output

#         return CausalLMOutput(
#             loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
#         )    
    
# #https://github.com/hirofumi0810/neural_sp/blob/000cd9dd657f83cd4883faf9ac48d0fcc40badb9/neural_sp/models/criterion.py#L110
# def kldiv_lsm_ctc(logits, ylens):
#     """Compute KL divergence loss for label smoothing of CTC and Transducer models.
#     Args:
#         logits (FloatTensor): `[B, T, vocab]`
#         ylens (IntTensor): `[B]`
#     Returns:
#         loss_mean (FloatTensor): `[1]`
#     """
#     bs, _, vocab = logits.size()

#     log_uniform = logits.new_zeros(logits.size()).fill_(math.log(1 / (vocab - 1)))
#     probs = torch.softmax(logits, dim=-1)
#     log_probs = torch.log_softmax(logits, dim=-1)
#     loss = torch.mul(probs, log_probs - log_uniform)
#     loss_mean = sum([loss[b, :ylens[b], :].sum() for b in range(bs)]) / ylens.sum()
#     return loss_mean

# #https://speechbrain.readthedocs.io/en/latest/_modules/speechbrain/nnet/losses.html#kldiv_loss
# def kldiv_loss(
#     log_probabilities,
#     targets,
#     length=None,
#     label_smoothing=0.0,
#     allowed_len_diff=3,
#     pad_idx=0,
#     reduction="mean",
# ):
#     """Computes the KL-divergence error at the batch level.
#     This loss applies label smoothing directly to the targets

#     Arguments
#     ---------
#     probabilities : torch.Tensor
#         The posterior probabilities of shape
#         [batch, prob] or [batch, frames, prob].
#     targets : torch.Tensor
#         The targets, of shape [batch] or [batch, frames].
#     length : torch.Tensor
#         Length of each utterance, if frame-level loss is desired.
#     allowed_len_diff : int
#         Length difference that will be tolerated before raising an exception.
#     reduction : str
#         Options are 'mean', 'batch', 'batchmean', 'sum'.
#         See pytorch for 'mean', 'sum'. The 'batch' option returns
#         one loss per item in the batch, 'batchmean' returns sum / batch size.

#     Example
#     -------
#     >>> probs = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
#     >>> kldiv_loss(torch.log(probs), torch.tensor([1, 1]))
#     tensor(1.2040)
#     """
#     if label_smoothing > 0:
#         if log_probabilities.dim() == 2:
#             log_probabilities = log_probabilities.unsqueeze(1)

#         bz, time, n_class = log_probabilities.shape
#         targets = targets.long().detach()

#         confidence = 1 - label_smoothing

#         log_probabilities = log_probabilities.view(-1, n_class)
#         targets = targets.view(-1)
#         with torch.no_grad():
#             true_distribution = log_probabilities.clone()
#             true_distribution.fill_(label_smoothing / (n_class - 1))
#             ignore = targets == pad_idx
#             targets = targets.masked_fill(ignore, 0)
#             true_distribution.scatter_(1, targets.unsqueeze(1), confidence)

#         loss = torch.nn.functional.kl_div(
#             log_probabilities, true_distribution, reduction="none"
#         )
#         loss = loss.masked_fill(ignore.unsqueeze(1), 0)

#         # return loss according to reduction specified
#         if reduction == "mean":
#             return loss.sum().mean()
#         elif reduction == "batchmean":
#             return loss.sum() / bz
#         elif reduction == "batch":
#             return loss.view(bz, -1).sum(1) / length
#         elif reduction == "sum":
#             return loss.sum()
#         else:
#             return loss
#     else:
#         return nll_loss(log_probabilities, targets, length, reduction=reduction)    
    
