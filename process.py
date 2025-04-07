# %% Imports
# Copyright (c) 2024 The HuggingFace Team
import copy

import evaluate
from datasets import load_from_disk
from google.cloud import translate_v2 as translate


# %% Functions
def translate_fil_to_eng(text: str, translate_client: translate.Client) -> str:
    result = translate_client.translate(
        text,
        target_language="en",
        source_language="tl",
    )

    return result["translatedText"]


def process_batch(examples, translate_client: translate.Client):
    examples["target-suggestion-back"] = [
        translate_fil_to_eng(text, translate_client) if text else "" for text in examples["target-suggestion"]
    ]

    examples["target-back"] = []
    for target_list in examples["target"]:
        target_back_list = copy.deepcopy(target_list)

        for item in target_back_list:
            if "value" in item and item["value"]:
                item["value"] = translate_fil_to_eng(item["value"], translate_client)

        examples["target-back"].append(target_back_list)

    return examples


# %% Run translation
# ds = load_dataset("data-is-better-together/MPEP_FILIPINO")["train"]
# translate_client = translate.Client()

# ds = ds.map(
#     lambda x: process_batch(x, translate_client),
#     batched=True,
#     batch_size=16,
#     desc="Backtranslating Filipino to English",
# )

# # Save the processed dataset
# ds.save_to_disk("MPEP_FILIPINO_with_backtranslation")

# # Display sample to verify
# print(ds[0]["source"])
# print(ds[0]["target-suggestion"])


# %% Get BLEU scores

ds = load_from_disk("./MPEP_FILIPINO_with_backtranslation")
metric = evaluate.load("bleu")


def _compute(prediction: str, reference: str) -> float:
    return metric.compute(predictions=[prediction], references=[reference])["bleu"]


def get_score(example: dict, metric: evaluate.Metric) -> dict:
    example["bleu-suggestion"] = _compute(
        example["target-suggestion-back"],
    )
    example["bleu"] = [_compute(d["value"], example["source"]) for d in example["target-back"]]

    return example


ds = ds.map(lambda x: get_score(x, metric))

# %%
bleu_diff = [[lst - ds["bleu-suggestion"][i] for lst in ds["bleu"][i]] for i in range(len(ds))]

# %%
