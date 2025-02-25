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
FiReCS, the first sentiment-annotated corpus of product and service reviews
involving Filipino-English code-switching. The data set is composed of 10,487
reviews with a fairly balanced number per sentiment class. Inter-annotator
agreement is high with a Kripendorffs’s α for ordinal metric of 0.83. Three
human annotators were tasked to manually label reviews according to three
polarity classes: Positive, Neutral, and Negative.

Paper link: https://link.springer.com/chapter/10.1007/978-981-99-8349-0_11
Dataset link: https://huggingface.co/datasets/ccosme/FiReCS
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


CHOICES = ["Negatibo", "Neutral", "Positibo"]

FILIPINO_FIRECS_TASK = [
    LightevalTaskConfig(
        name=f"firecs_fil_{formulation.name.lower()}",
        hf_subset="default",
        prompt_function=get_mcq_prompt_function(
            Language.TAGALOG,
            lambda line: {
                "question": f"Ano ang damdamin o sentimiyento ng sumusunod na pangungusap: {line['review']}",
                "choices": CHOICES,
                "gold_idx": int(line["label"]),
            },
        ),
        hf_repo="ccosme/FiReCS",
        metric=get_metrics_for_formulation(
            formulation,
            [
                loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
                loglikelihood_acc_metric(normalization=LogProbCharNorm()),
                loglikelihood_acc_metric(normalization=LogProbPMINorm()),
            ],
        ),
        hf_avail_splits=["train", "test"],
        evaluation_splits=["train"],
        few_shots_split="train",
        few_shots_select="random",
        suite=["filbench"],
        generation_size=-1,
        trust_dataset=True,
        version=0,
    )
    for formulation in [MCFFormulation(), HybridFormulation()]
]
