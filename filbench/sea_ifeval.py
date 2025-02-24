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
SEA-IFEval evaluates a model's ability to adhere to constraints provided in the
prompt, for example beginning a response with a specific word/phrase or
answering with a certain number of sections. It is based on IFEval and was
manually translated by native speakers for Indonesian, Javanese, Sundanese,
Thai, Tagalog, and Vietnamese.
"""

from lighteval.tasks.extended.ifeval.main import ifeval_metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


# Very specific task where there are no precise outputs but instead we test if the format obeys rules
# Reference: https://github.com/huggingface/lighteval/blob/ebb7377b39a48ab0691e6fbd9dea57e9fe290a7e/src/lighteval/tasks/extended/ifeval/main.py#L38
def ifeval_prompt(line, task_name: str = None):
    query = line["prompts"][0]["text"]
    instruction_id_list = [line["metadata"].get("subcategory")]

    return Doc(
        task_name=task_name,
        query=query,
        choices=[""],
        gold_index=0,
        instruction="",
        specific={"instruction_id_list": instruction_id_list, "kwargs": line["kwargs"]},
    )


task = LightevalTaskConfig(
    name="seaifeval_tgl",
    prompt_function=ifeval_prompt,
    suite=("filbench",),
    hf_repo="aisingapore/instruction_following-ifeval",
    hf_subset="default",
    hf_avail_splits=["tl"],
    evaluation_splits=["tl"],
    few_shots_split="tl",
    metric=[ifeval_metrics],
    few_shots_select="random_sampling",
    generation_size=1280,  # https://github.com/huggingface/lighteval/blob/ebb7377b39a48ab0691e6fbd9dea57e9fe290a7e/src/lighteval/tasks/extended/ifeval/main.py#L154
    stop_sequence=[],  # no stop sequence, will use eot token,
    version="0.1",
)


FILIPINO_SEAIFEVAL_TASKS = [task]
