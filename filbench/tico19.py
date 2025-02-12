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
Translation Initiative for COvid-19 (TICO-19) have wtest and development
data available to AI and MT researchers in 35 different languages in order to
foster the development of tools and resources for improving access to
information about COVID-19 in these languages. In addition to 9 high-resourced,
"pivot" languages, the team is targeting 26 lesser resourced languages, in
particular languages of Africa, South Asia and South-East Asia, whose
populations may be the most vulnerable to the spread of the virus. The same data
is translated into all of the languages represented, meaning that testing or
development can be done for any pairing of languages in the set. Further, the
team is converting the test and development data into translation memories
(TMXs) that can be used by localizers from and to any of the languages.

Paper link: https://arxiv.org/abs/2007.01788
Dataset link: https://huggingface.co/datasets/gmnlp/tico19/viewer/en-tl
"""

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.templates.translation import get_translation_prompt_function
from lighteval.tasks.templates.utils.formulation import CFFormulation
from lighteval.utils.language import Language


FILIPINO_TICO19_TASKS = [
    LightevalTaskConfig(
        name="tico19_tgl",
        prompt_function=get_translation_prompt_function(
            source_language=Language.ENGLISH,
            target_language=Language.TAGALOG,
            adapter=lambda line: {
                "source_text": line["sourceString"],
                "target_text": line["targetString"],
            },
            formulation=CFFormulation(),
        ),
        suite=("filbench",),
        hf_repo="gmnlp/tico19",
        hf_subset="en-tl",
        metric=[
            Metrics.rougeL,
            Metrics.bleu,
            Metrics.bleurt,
            Metrics.chrf,
            Metrics.ter,
        ],
        hf_avail_splits=["test", "validation"],
        evaluation_splits=["validation"],
        few_shots_split=["validation"],
        few_shots_select="random",
        trust_dataset=True,
    )
]
