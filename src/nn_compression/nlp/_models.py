from enum import Enum
from transformers import GPT2LMHeadModel, GPTNeoXForCausalLM
from ._datasets import Wikitext
import torch


class LanguageModel(str, Enum):
    GPT2 = "gpt2"
    GPT2_XL = "gpt2-xl"
    PYTHIA_70M = "pythia-70m"
    PYTHIA_1B = "pythia-1b"

    def load(self) -> torch.nn.Module:
        if self.value == "gpt2" or self.value == "gpt2-xl":
            return GPT2LMHeadModel.from_pretrained(self.value)  # type: ignore
        elif self.value.startswith("pythia"):
            return GPTNeoXForCausalLM.from_pretrained(  # type: ignore
                f"EleutherAI/{self.value}-deduped",
            )
        else:
            raise ValueError(f"Unknown model {self.value}")

    @staticmethod
    def from_string(s: str):
        return LanguageModel(s)

    def get_dataset(self, shuffle=True, batch_size=1):
        return Wikitext(self, shuffle, batch_size)

    def filter_fn(self):
        if self.value.startswith("pythia"):
            return lambda n: n.startswith("gpt_neox.layers")
        if self.value.startswith("gpt"):
            return lambda n: n.startswith("transformer.h")
        else:
            raise NotImplementedError(
                f"Didn't implement filter_fn for {self.value} (yet)."
            )
