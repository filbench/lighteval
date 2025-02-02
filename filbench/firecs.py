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
FiReCS, the first sentiment-annotated corpus of product and service reviews
involving Filipino-English code-switching. The data set is composed of 10,487
reviews with a fairly balanced number per sentiment class. Inter-annotator
agreement is high with a Kripendorffs’s α for ordinal metric of 0.83. Three
human annotators were tasked to manually label reviews according to three
polarity classes: Positive, Neutral, and Negative.

Paper link: https://link.springer.com/chapter/10.1007/978-981-99-8349-0_11
Dataset link: https://huggingface.co/datasets/ccosme/FiReCS
"""

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


def filipino_firecs_pfn(line, task_name: str = None) -> Doc:
    instruction = "Ano ang damdamin o sentimyento ng sumusunod na pangungusap. Piliin ang numero ng tamang sagot:\n\n"
    choices = []
    valid_keys = []

    for key in ["0", "1", "2"]:
        option = line.get(f"sol{key}")
        if option:
            choices.append(option)
            valid_keys.append(key)

    answer_index = int(line.get("label"))
    query = f"{instruction}{line['review']}\n"
    query += "".join([f"{key}. {choice}\n" for key, choice in zip(valid_keys, choices)])
    query += "Sagot:"
    return Doc(
        task_name=task_name,
        query=query,
        choices=valid_keys,
        gold_index=answer_index,
        instruction=instruction,
    )


FILIPINO_FIRECS_TASK = [
    LightevalTaskConfig(
        name="firecs",
        hf_subset="default",
        prompt_function=filipino_firecs_pfn,
        hf_repo="UD-Filipino/FiReCS",
        metric=[Metrics.loglikelihood_acc_norm],
        hf_avail_splits=["test"],
        suite=["filbench"],
        generation_size=-1,
        trust_dataset=True,
        version=0,
    )
]
