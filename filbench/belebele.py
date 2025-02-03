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
Belebele is a multiple-choice machine reading comprehension (MRC) dataset
spanning 122 language variants. This dataset enables the evaluation of mono- and
multi-lingual models in high-, medium-, and low-resource languages. Each
question has four multiple-choice answers and is linked to a short passage from
the FLORES-200 dataset. The human annotation procedure was carefully curated to
create questions that discriminate between different levels of generalizable
language comprehension and is reinforced by extensive quality checks. While all
questions directly relate to the passage, the English dataset on its own proves
difficult enough to challenge state-of-the-art language models. Being fully
parallel, this dataset enables direct comparison of model performance across all
languages. Belebele opens up new avenues for evaluating and analyzing the
multilingual abilities of language models and NLP systems.

Paper link: https://aclanthology.org/2024.acl-long.44/
Dataset link: https://huggingface.co/datasets/facebook/belebele
"""

from langcodes import Language as LangCodeLanguage

from lighteval.metrics.dynamic_metrics import loglikelihood_acc_metric
from lighteval.metrics.normalizations import (
    LogProbCharNorm,
    LogProbTokenNorm,
)
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.multilingual.utils.task_utils import get_metrics_for_formulation
from lighteval.tasks.templates.multichoice import get_mcq_prompt_function
from lighteval.tasks.templates.utils.formulation import (
    CFFormulation,
    HybridFormulation,
    MCFFormulation,
)
from lighteval.utils.language import iso_639_3_ind_to_iso_639_3_macro


FILIPINO_BELEBELE_TASKS = [
    LightevalTaskConfig(
        name=f"belebele_{iso_639_3_ind_to_iso_639_3_macro[LangCodeLanguage.get(language).to_alpha3()]}_{formulation.name.lower()}",
        prompt_function=get_mcq_prompt_function(
            iso_639_3_ind_to_iso_639_3_macro[LangCodeLanguage.get(language).to_alpha3()],
            lambda line: {
                "question": line["question"],
                "context": line["flores_passage"],
                "choices": [line[f"mc_answer{i}"] for i in range(1, 5)],
                "gold_idx": int(line["correct_answer_num"]) - 1,
            },
            formulation=formulation,
        ),
        suite=("filbench",),
        hf_repo="facebook/belebele",
        hf_subset=language,
        evaluation_splits=("test",),
        hf_avail_splits=["test"],
        metric=get_metrics_for_formulation(
            formulation,
            [
                loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
                loglikelihood_acc_metric(normalization=LogProbCharNorm()),
            ],
        ),
    )
    for formulation in [MCFFormulation(), CFFormulation(), HybridFormulation()]
    for language in ["tgl_Latn", "ceb_Latn"]
]
