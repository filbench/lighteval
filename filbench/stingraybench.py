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
StingrayBench demonstrates using false friends -- words that are
orthographically similar but have completely different meanings in two
languages -- as a possible approach to pinpoint the limitation of cross-lingual
sense disambiguation in LLMs.

Paper link: https://arxiv.org/abs/2410.21573
Dataset link: https://huggingface.co/datasets/StingrayBench/StingrayBench
"""

from typing import Any

from lighteval.metrics.dynamic_metrics import loglikelihood_acc_metric
from lighteval.metrics.normalizations import (
    LogProbCharNorm,
    LogProbPMINorm,
    LogProbTokenNorm,
)
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.multilingual.utils.task_utils import get_metrics_for_formulation
from lighteval.tasks.templates.multichoice import get_mcq_prompt_function
from lighteval.tasks.templates.utils.formulation import (
    HybridFormulation,
    MCFFormulation,
)
from lighteval.utils.language import Language


def prepare_stingray_correctness(line: dict[str, str]) -> dict[str, Any]:
    # lang2 is Tagalog
    word = line["word"]
    sentence = line["lang2_sentence"]
    question = f"Is the usage of {word} in this sentence correct? \n{sentence}"
    choices = ["Yes", "No"]
    gold_idx = choices.index(line["usage_correctness2_lang2_answer"])
    return {"question": question, "choices": choices, "gold_idx": gold_idx}


def prepare_stingray_semantic_appropriateness(line: dict[str, str]) -> dict[str, Any]:
    lang1 = line["lang1_sentence"]
    lang2 = line["lang2_sentence"]
    question = "Which sentence is more semantically appropriate?"
    choices = [lang1, lang2, "Both"]
    choice_letters = ["A", "B", "C"]
    gold_idx = choice_letters.index(line["semantic_appropriate_answer"])
    return {"question": question, "choices": choices, "gold_idx": gold_idx}


FILIPINO_STINGRAY_CORRECTNESS_TASKS = [
    LightevalTaskConfig(
        name=f"stingraybench_correctness_tgl_{formulation.name.lower()}",
        prompt_function=get_mcq_prompt_function(
            Language.ENGLISH,  # the orig instruction is in English, so we replicate it.
            adapter=prepare_stingray_correctness,
            formulation=formulation,
        ),
        suite=("filbench",),
        hf_subset="id_tl",
        hf_repo="StingrayBench/StingrayBench",
        metric=get_metrics_for_formulation(
            formulation,
            [
                loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
                loglikelihood_acc_metric(normalization=LogProbCharNorm()),
                loglikelihood_acc_metric(normalization=LogProbPMINorm()),
            ],
        ),
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        few_shots_split="test",
        few_shots_select="random",
        generation_size=-1,
        trust_dataset=True,
        version=0,
    )
    for formulation in [MCFFormulation(), HybridFormulation()]
]

FILIPINO_STINGRAY_SEMANTIC_TAKS = [
    LightevalTaskConfig(
        name=f"stingraybench_semantic_appropriateness_tgl_{formulation.name.lower()}",
        prompt_function=get_mcq_prompt_function(
            Language.ENGLISH,  # the orig instruction is in English, so we replicate it.
            adapter=prepare_stingray_semantic_appropriateness,
            formulation=formulation,
        ),
        suite=("filbench",),
        hf_subset="id_tl",
        hf_repo="StingrayBench/StingrayBench",
        metric=get_metrics_for_formulation(
            formulation,
            [
                loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
                loglikelihood_acc_metric(normalization=LogProbCharNorm()),
                loglikelihood_acc_metric(normalization=LogProbPMINorm()),
            ],
        ),
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        few_shots_split="test",
        few_shots_select="random",
        generation_size=-1,
        trust_dataset=True,
        version=0,
    )
    for formulation in [MCFFormulation(), HybridFormulation()]
]

FILIPINO_STINGRAY_TASKS = FILIPINO_STINGRAY_SEMANTIC_TAKS + FILIPINO_STINGRAY_CORRECTNESS_TASKS
