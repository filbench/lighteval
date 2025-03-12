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
The Readability benchmark is from a study that looked into creating the first
baseline readability model for Cebuano.  The authors extracted traditional or
surface-based features, syllable patterns based from Cebuano's documented
orthography, and neural embeddings from the multilingual BERT model.  Results
show that the use of the first two handcrafted linguistic features obtained the
best performance trained on an optimized Random Forest model with approximately
87% across all metrics.
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


CHOICES = ["Grade 1", "Grade 2", "Grade 3"]
INSTRUCTION = """
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
                "question": INSTRUCTION + line["text"],
                "choices": CHOICES,
                "gold_idx": CHOICES.index(f"Grade {line['label']}"),
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
