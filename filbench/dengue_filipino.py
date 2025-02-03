# MIT License

# Copyright (c) 2024 The HuggingFace Team

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# ruff: noqa: F405, F403, F401
"""
The Dengue Filipino dataset is a multiclass classification dataset of tweets
related to dengue. 5,015 tweets were manually labeled as possibly part of any
of the following classes: absent, dengue, health, mosquito or sick.

Paper link: https://ieeexplore.ieee.org/document/8459963
Dataset link: https://huggingface.co/datasets/jcblaise/dengue_filipino
"""
from collections import OrderedDict

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


dengue_filipino_subsets = {
    "absent": "pagiging absent",
    "dengue": "dengue",
    "health": "kalusugan",
    "mosquito": "lamok",
    "sick": "sakit",
}


def filipino_dengue_pfn(line, task_name: str) -> Doc:
    subset = task_name.split(":")[-1]
    subset_keyword = dengue_filipino_subsets[subset]

    instruction = f"Tungkol ba sa {subset_keyword} ang sumusunod na pangungusap? Piliin ang tamang sagot:\n\n"
    choices: dict[str, str] = OrderedDict({"A": "Hindi", "B": "Oo"})

    answer_index = int(line.get(subset))
    query = f"{instruction}{line['text']}\n"
    query += "".join([f"{key}. {choice}\n" for key, choice in choices.items()])
    query += "Sagot:"
    return Doc(
        task_name=task_name,
        query=query,
        choices=list(choices.keys()),
        gold_index=answer_index,
        instruction=instruction,
    )


FILIPINO_DENGUE_TASKS = [
    LightevalTaskConfig(
        name=f"dengue_filipino_fil:{subset}",
        hf_subset="default",
        prompt_function=filipino_dengue_pfn,
        hf_repo="jcblaise/dengue_filipino",
        metric=[Metrics.loglikelihood_acc_norm],
        hf_avail_splits=["train", "test", "validation"],
        evaluation_splits=["train"],
        few_shots_split="train",
        few_shots_select="random",
        suite=("filbench",),
        generation_size=-1,
        trust_dataset=True,
        version=0,
    )
    for subset in dengue_filipino_subsets
]
