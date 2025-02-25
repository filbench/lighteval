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
SIB-200 is the largest publicly available topic classification dataset based on Flores-200 covering 205 languages and dialects.
The train/validation/test sets are available for all the 205 languages.

Paper link: https://aclanthology.org/2024.eacl-long.14/
Dataset link: https://huggingface.co/datasets/Davlan/sib200
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


CHOICES = [
    "geography",
    "science/technology",
    "entertainment",
    "travel",
    "sports",
    "health",
    "politics",
]


def get_instruction(language: Language) -> str:
    if language == Language.CEBUANO:
        return "Mahitungod sa unsa ang mosunod nga teksto?\n"
    if language == Language.TAGALOG:
        return "Tungkol saan ang sumusunod na pangungusap?\n"


def create_task(language: Language, formulation):
    return LightevalTaskConfig(
        name=f"sib200_{language.value}_{formulation.name.lower()}",
        prompt_function=get_mcq_prompt_function(
            language,
            lambda line: {
                "question": get_instruction(language) + line["text"],
                "choices": CHOICES,
                "gold_idx": CHOICES.index(line["category"]),
            },
            formulation=formulation,
        ),
        suite=("filbench",),
        hf_subset=f"{language.value}_Latn",
        hf_repo="Davlan/sib200",
        metric=get_metrics_for_formulation(
            formulation,
            [
                loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
                loglikelihood_acc_metric(normalization=LogProbCharNorm()),
                loglikelihood_acc_metric(normalization=LogProbPMINorm()),
            ],
        ),
        hf_avail_splits=["test", "validation"],
        evaluation_splits=["validation"],
        few_shots_split="validation",
        few_shots_select="random",
        generation_size=-1,
        trust_dataset=True,
        version=0,
    )


FILIPINO_SIB_TASKS = [
    create_task(language, formulation)
    for language in [Language.TAGALOG, Language.CEBUANO]
    for formulation in [MCFFormulation(), HybridFormulation()]
]
