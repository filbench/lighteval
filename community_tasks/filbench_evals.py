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
Custom evaluation tasks for lighteval.

This file generally creates just a TASKS_TABLE and TASKS_GROUPS which are then imported by LightEval.
"""

from filbench.belebele import FILIPINO_BELEBELE_TASKS
from filbench.cebuaner import FILIPINO_CEBUANER_TASKS
from filbench.cebuano_readability import FILIPINO_READABILITY_TASKS
from filbench.dengue_filipino import FILIPINO_DENGUE_TASKS
from filbench.firecs import FILIPINO_FIRECS_TASK
from filbench.global_mmlu import FILIPINO_GLOBAL_MMLU_TASKS
from filbench.include import FILIPINO_INCLUDE_TASKS
from filbench.kalahi import FILIPINO_KALAHI_TASKS
from filbench.newsph_nli import FILIPINO_NEWSPH_NLI_TASKS
from filbench.ntrex128 import FILIPINO_NTREX_TASK
from filbench.sea_ifeval import FILIPINO_SEAIFEVAL_TASKS
from filbench.sib200 import FILIPINO_SIB_TASKS
from filbench.stingraybench import FILIPINO_STINGRAY_TASKS
from filbench.tatoeba import FILIPINO_TATOEBA_TASKS
from filbench.tico19 import FILIPINO_TICO19_TASKS
from filbench.tlunified_ner import FILIPINO_TLUNIFIED_NER_TASK
from filbench.universal_ner import FILIPINO_UNIVERSALNER_TASKS
from lighteval.tasks.lighteval_task import LightevalTaskConfig


TASKS_TABLE: list[LightevalTaskConfig] = (
    FILIPINO_GLOBAL_MMLU_TASKS
    + FILIPINO_FIRECS_TASK
    + FILIPINO_SIB_TASKS
    + FILIPINO_BELEBELE_TASKS
    + FILIPINO_NEWSPH_NLI_TASKS
    + FILIPINO_INCLUDE_TASKS
    + FILIPINO_DENGUE_TASKS
    + FILIPINO_NTREX_TASK
    + FILIPINO_TICO19_TASKS
    + FILIPINO_STINGRAY_TASKS
    + FILIPINO_TATOEBA_TASKS
    + FILIPINO_SEAIFEVAL_TASKS
    + FILIPINO_TLUNIFIED_NER_TASK
    + FILIPINO_UNIVERSALNER_TASKS
    + FILIPINO_CEBUANER_TASKS
    + FILIPINO_KALAHI_TASKS
    + FILIPINO_READABILITY_TASKS
)
