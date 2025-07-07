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

from langcodes import Language as LangCodeLanguage

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
    CFFormulation,
    HybridFormulation,
    MCFFormulation,
)
from lighteval.utils.language import Language, iso_639_3_ind_to_iso_639_3_macro


# Balita NLP
FILIPINO_BALITA_TASKS = [
    LightevalTaskConfig(
        name=f"balita_tgl_{formulation.name.lower()}",
        prompt_function=get_mcq_prompt_function(
            language=Language.TAGALOG,
            adapter=lambda line: {
                "question": "Alin sa mga titlulong nakalista sa ibaba ang pinaka-angkop para sa teksto?",
                "context": f"Teksto: {line['title_choice_first_paragraph']}",
                "choices": line["title_choices"],
                "gold_idx": line["title_choice_gold_idx"],
            },
            formulation=formulation,
        ),
        suite=("filbench",),
        hf_repo="LanceBunag/BalitaNLP",
        hf_subset="no-image",
        hf_avail_splits=["train", "validation", "test"],
        evaluation_splits=("validation", "test"),
        metric=get_metrics_for_formulation(
            formulation,
            [
                loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
                loglikelihood_acc_metric(normalization=LogProbCharNorm()),
            ],
        ),
    )
    for formulation in [MCFFormulation(), HybridFormulation()]
]

# Belebele
FILIPINO_BELEBELE_TASKS = [
    LightevalTaskConfig(
        name=f"belebele_{LangCodeLanguage.get(language).to_alpha3()}_{formulation.name.lower()}",
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

# CebuaNER
cebuaner_choices = ["PERSON", "ORGANIZATION", "LOCATION", "OTHER"]
cebuaner_answer_idx = ["A", "B", "C", "D"]
question = "Unsa ang ginganlan nga named-entity sa pulong '{entity}' niini nga sentence: {text}"
FILIPINO_CEBUANER_TASKS = [
    LightevalTaskConfig(
        name=f"cebuaner_ceb_{formulation.name.lower()}",
        hf_subset="default",
        prompt_function=get_mcq_prompt_function(
            Language.CEBUANO,
            lambda line: {
                "question": question.format(entity=line["entity"], text=line["text"]),
                "choices": cebuaner_choices,
                "gold_idx": cebuaner_answer_idx.index(line["answer"]),
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

# Cebuano Readability
cebuano_readability_choices = ["Grade 1", "Grade 2", "Grade 3"]
cebuano_readability_instruction = """
Unsa ang angay nga lebel sa grado alang sa mosunod nga teksto?

Grade 1 - ang teksto mahimong basahon sa usa ka tawo tali sa edad nga 6-7.
Grade 2 - ang teksto mahimong basahon sa usa ka tawo tali sa edad nga 7-8.
Grade 3 - ang teksto mahimong basahon sa usa ka tawo tali sa edad nga 8-9.
"""
FILIPINO_READABILITY_TASKS = [
    LightevalTaskConfig(
        name=f"readability_ceb_{formulation.name.lower()}",
        prompt_function=get_mcq_prompt_function(
            Language.CEBUANO,
            lambda line: {
                "question": cebuano_readability_instruction + line["text"],
                "choices": cebuano_readability_choices,
                "gold_idx": cebuano_readability_choices.index(f"Grade {line['label']}"),
            },
            formulation=formulation,
        ),
        suite=("filbench",),
        hf_subset="default",
        hf_repo="UD-Filipino/cebuano-readability",
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

TASKS_TABLE: list[LightevalTaskConfig] = (
    FILIPINO_BALITA_TASKS + FILIPINO_BELEBELE_TASKS + FILIPINO_CEBUANER_TASKS + FILIPINO_READABILITY_TASKS
)
