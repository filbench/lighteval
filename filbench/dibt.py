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
Manually annotated translation dataset for English to Tagalog (Filipino) from the
FilBench team.

Dataset link: https://huggingface.co/datasets/gmnlp/tico19/viewer/en-tl
"""

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.templates.translation import get_translation_prompt_function
from lighteval.tasks.templates.utils.formulation import CFFormulation
from lighteval.utils.language import Language


FILIPINO_DIBT_TASKS = [
    LightevalTaskConfig(
        name="dibt_translation_tgl",
        prompt_function=get_translation_prompt_function(
            source_language=Language.ENGLISH,
            target_language=Language.TAGALOG,
            adapter=lambda line: {
                "source_text": line["source"],
                "target_text": [entry["value"] for entry in line["target"]],
            },
            formulation=CFFormulation(),
        ),
        suite=("filbench",),
        hf_repo="data-is-better-together/MPEP_FILIPINO",
        metric=[
            Metrics.rougeL,
            Metrics.bleu,
            Metrics.bleurt,
            Metrics.chrf,
            Metrics.ter,
        ],
        evaluation_splits=["train"],
        trust_dataset=True,
        generation_size=64,
    )
]
