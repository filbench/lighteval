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
INCLUDE is a knowledge- and reasoning-centric benchmark with evalutation
coverage among 44 languages made up of multiple-choice questions extracted
from academic and professional exams.

Paper link: https://arxiv.org/abs/2411.19799
Dataset link: https://huggingface.co/datasets/CohereForAI/include-base-44

Implementation based on: https://github.com/huggingface/lighteval/blob/d332207bf65d70d3a1fe0538af91565d60cf47dd/src/lighteval/tasks/multilingual/tasks.py#L1716
"""
from functools import partial

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


FILIPINO_INCLUDE_TASKS = [
    LightevalTaskConfig(
        name=f"include_{language.value}_{formulation.name.lower()}:{subset}",
        prompt_function=get_mcq_prompt_function(
            language,
            lambda line: {"question": line["question"], "choices": line["choices"], "gold_idx": line["answer"]},
            formulation=formulation,
        ),
        suite=("filbench",),
        hf_subset="Tagalog",
        hf_repo="CohereForAI/include-base-44",
        hf_filter=partial(lambda subset, x: x["subject"].replace(" ", "_").lower() == subset, subset),
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
    for subset in ["culturology", "history", "language", "driving_license"]
    for language in [Language.TAGALOG]
    for formulation in [MCFFormulation(), HybridFormulation()]
]
