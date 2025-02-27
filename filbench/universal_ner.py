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
We introduce Universal NER (UNER), an open, community-driven project to develop
gold-standard NER benchmarks in many languages. The overarching goal of UNER is
to provide high-quality, cross-lingually consistent annotations to facilitate
and standardize multilingual NER research. UNER v1 contains 18 datasets
annotated with named entities in a cross-lingual consistent schema across 12
diverse languages. In this paper, we detail the dataset creation and composition
of UNER; we also provide initial modeling baselines on both in-language and
cross-lingual learning settings. We release the data, code, and fitted models to
the public.

Paper: https://aclanthology.org/2024.naacl-long.243/
Dataset: https://huggingface.co/datasets/universalner/universal_ner
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


CHOICES = ["PERSON", "ORGANIZATION", "LOCATION"]
ANSWER_IDX = ["A", "B", "C"]


def create_task(language: Language, formulation):
    if language == Language.CEBUANO:
        question = "Unsa ang ginganlan nga named-entity sa pulong '{entity}' niini nga sentence: {text}"
    if language == Language.TAGALOG:
        question = "Ano ang named-entity ng salitang '{entity}' sa pangungusap na ito: {text}"

    return LightevalTaskConfig(
        name=f"universalner_{language.value}_{formulation.name.lower()}",
        hf_subset=language.value,
        prompt_function=get_mcq_prompt_function(
            language,
            lambda line: {
                "question": question.format(entity=line["entity"], text=line["text"]),
                "choices": CHOICES,
                "gold_idx": ANSWER_IDX.index(line["answer"]),
            },
            formulation=formulation,
        ),
        hf_repo="UD-Filipino/universalner-instruction",
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        few_shots_split="test",
        few_shots_select="random",
        suite=["filbench"],
        generation_size=16,
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


FILIPINO_UNIVERSALNER_TASKS = [
    create_task(language, formulation)
    for language in [Language.CEBUANO, Language.TAGALOG]
    for formulation in [MCFFormulation(), HybridFormulation()]
]
