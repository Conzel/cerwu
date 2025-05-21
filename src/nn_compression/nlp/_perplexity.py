from typing import Literal, Optional
import torch
from tqdm import tqdm
import torch.nn as nn
from transformers.tokenization_utils_base import BatchEncoding


def perplexity(
    model: nn.Module,
    encodings: BatchEncoding,
    stride: int,
    max_steps: Optional[int] = None,
    device: Literal["cpu", "cuda", "mps"] = "cpu",
    verbose: bool = False,
    predict_runtime: bool = False,
) -> float:
    """Calculate the perplexity of a model on a given dataset.
    The perplexity is defined as the exponential of the average negative log likelihood of the model.
    Perplexity is calculated through a sliding window strategy, see
    https://huggingface.co/docs/transformers/en/perplexity.
    """
    mode_before = model.training
    model.train(False)
    model.to(device)
    if device == "mps":
        print(
            "MPS for perplexity can have errors, see https://github.com/huggingface/transformers/issues/32005"
        )
    # max_length = model.config.n_positions
    max_length = 1024
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    i = 0
    for begin_loc in tqdm(
        range(0, seq_len, stride), total=max_steps, disable=not verbose
    ):
        if max_steps is not None and i >= max_steps:
            break
        i += 1

        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone().to(device)
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)

            # loss is calculated using CrossEntropyLoss which averages over valid labels
            # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
            # to the left by 1.
            neg_log_likelihood = outputs["loss"]

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    model.train(mode_before)
    return torch.exp(torch.stack(nlls).mean()).detach().cpu().item()
