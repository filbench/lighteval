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

"""
We introduce CebuaNER, a new baseline model for named entity recognition (NER)
in the Cebuano language. Cebuano is the second most-used native language in the
Philippines with over 20 million speakers.

Paper: https://arxiv.org/abs/2310.00679
Dataset: https://huggingface.co/datasets/SEACrowd/cebuaner
"""

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


CHOICES = ["PERSON", "ORGANIZATION", "LOCATION", "OTHER"]
ANSWER_IDX = ["A", "B", "C", "D"]

question = "Unsa ang ginganlan nga named-entity sa pulong '{entity}' niini nga sentence: {text}"

FILIPINO_CEBUANER_TASKS = [
    LightevalTaskConfig(
        name=f"cebuaner_ceb_{formulation.name.lower()}",
        hf_subset="default",
        prompt_function=get_mcq_prompt_function(
            Language.CEBUANO,
            lambda line: {
                "question": question.format(entity=line["entity"], text=line["text"]),
                "choices": CHOICES,
                "gold_idx": ANSWER_IDX.index(line["answer"]),
            },
            formulation=formulation,
        ),
        hf_repo="UD-Filipino/cebuaner-instruction",
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        few_shots_split="test",
        few_shots_select="random",
        suite=["filbench"],
        generation_size=-1,
        trust_dataset=True,
        metric=get_metrics_for_formulation(
            formulation,
            [
                loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
                loglikelihood_acc_metric(normalization=LogProbCharNorm()),
                loglikelihood_acc_metric(normalization=LogProbPMINorm()),
            ],
        ),
        version=0,
    )
    for formulation in [MCFFormulation(), HybridFormulation()]
]
