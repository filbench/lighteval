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

from lighteval.metrics.dynamic_metrics import loglikelihood_acc_metric
from lighteval.metrics.normalizations import LogProbCharNorm, LogProbTokenNorm
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.tasks.templates.multichoice import get_mcq_prompt_function
from lighteval.tasks.templates.utils.formulation import (
    HybridFormulation,
    MCFFormulation,
)
from lighteval.utils.language import Language


FILIPINO_KALAHI_TASKS = [
    LightevalTaskConfig(
        name=f"kalahi_tgl_{formulation.name.lower()}",
        suite=["filbench"],
        prompt_function=get_mcq_prompt_function(
            language=Language.TAGALOG,
            adapter=lambda line: {
                "question": line["prompt"],
                "choices": [line["best_answer"]] + line["irrelevant_answers"],
                "gold_idx": 0,
            },
            formulation=formulation,
        ),
        hf_repo="aisingapore/kalahi",
        hf_subset="default",
        evaluation_splits=["train"],
        metric=[
            loglikelihood_acc_metric(normalization=None),
            loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
            loglikelihood_acc_metric(normalization=LogProbCharNorm()),
        ],
    )
    for formulation in [HybridFormulation(), MCFFormulation()]
]
