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
This dataset contains the annotated TLUnified corpora from Cruz and Cheng
(2021). It is a curated sample of around 7,000 documents for the named entity
recognition (NER) task. The majority of the corpus are news reports in Tagalog,
resembling the domain of the original ConLL 2003. There are three entity types:
Person (PER), Organization (ORG), and Location (LOC).

Paper: https://aclanthology.org/2023.sealp-1.2/
Dataset: https://huggingface.co/datasets/ljvmiranda921/tlunified-ner
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


FILIPINO_TLUNIFIED_NER_TASK = [
    LightevalTaskConfig(
        name=f"tlunifiedner_tgl_{formulation.name.lower()}",
        hf_subset="instruction",
        prompt_function=get_mcq_prompt_function(
            Language.TAGALOG,
            lambda line: {
                "question": f"Ano ang named-entity ng salitang '{line['entity']}' sa pangungusap na ito: {line['text']}",
                "choices": CHOICES,
                "gold_idx": ANSWER_IDX.index(line["answer"]),
            },
            formulation=formulation,
        ),
        hf_repo="ljvmiranda921/tlunified-ner",
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
