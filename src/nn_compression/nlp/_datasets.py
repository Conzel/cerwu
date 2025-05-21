from typing import Literal, Optional
from transformers import GPT2TokenizerFast, BatchEncoding, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import Dataset
from dataclasses import dataclass
import torch
from torch.utils.data import DataLoader, TensorDataset
from ._perplexity import perplexity


def sequence_to_dataloader(
    encodings: torch.Tensor,
    max_size: int = 1024,
    batch_size: int = 1,
    return_labels: bool = True,
    max_samples: Optional[int] = None,
) -> DataLoader:
    """Assumes that the encodings are integer tensors with shape [1, n].
    The last sentence is clipped if it is shorter than max_size (#TODO)

    Returns a DataLoader with the encodings split into non-overlapping chunks of size max_size.
    """
    tensors = []
    for i in range(0, encodings.size(1), max_size):
        if i + max_size >= encodings.size(1):
            break
        if max_samples is not None and len(tensors) >= max_samples:
            break
        tensors.append(encodings[0, i : i + max_size])
    xs = torch.stack(tensors)
    if return_labels:
        ds = TensorDataset(xs, xs.clone())
    else:
        ds = TensorDataset(xs)
    return DataLoader(ds, batch_size=batch_size)


def wikitext(
    model,
    split: Literal["train", "test"],
) -> BatchEncoding:
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)

    # this might have to change if we want to use other models, but
    # for now its just GPT2
    if model.value == "gpt2" or model == "gpt2-xl":
        tokenizer = GPT2TokenizerFast.from_pretrained(model.value)
    elif model.value.startswith("pythia"):
        tokenizer = AutoTokenizer.from_pretrained(f"EleutherAI/{model.value}-deduped")
    encodings = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt")  # type: ignore

    return encodings


@dataclass
class Wikitext(Dataset):
    train_dataset: BatchEncoding
    test_dataset: BatchEncoding
    train_dataloader: DataLoader
    test_dataloader: DataLoader

    def __init__(self, model, shuffle=True, batch_size: int = 1) -> None:
        self.train_dataset = wikitext(model, "train")
        self.test_dataset = wikitext(model, "test")
        self.train_dataloader = sequence_to_dataloader(self.train_dataset["input_ids"], batch_size=batch_size)  # type: ignore
        self.test_dataloader = sequence_to_dataloader(self.test_dataset["input_ids"], batch_size=batch_size)  # type: ignore

    def evaluate(
        self,
        model,
        nbatches: int = 1,
        device: Literal["cpu", "mps", "cuda"] = "cpu",
        predict_runtime=False,
    ):
        return perplexity(
            model,
            self.test_dataset,
            512,
            nbatches,
            device=device,
            verbose=predict_runtime,
        )
