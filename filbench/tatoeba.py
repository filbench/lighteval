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
The Tatoeba Translation Challenge is a multilingual data set of machine
translation benchmarks derived from user-contributed translations collected by
Tatoeba.org and provided as parallel corpus from OPUS. This dataset includes
test and development data sorted by language pair. It includes test sets for
hundreds of language pairs and is continuously updated. Please, check the
version number tag to refer to the release that your are using.

Paper link: https://aclanthology.org/2020.wmt-1.139/
Dataset link: https://huggingface.co/datasets/Helsinki-NLP/tatoeba_mt
"""

from imp import get_tag

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.templates.translation import get_translation_prompt_function
from lighteval.tasks.templates.utils.formulation import CFFormulation
from lighteval.utils.language import Language


# We follow the original translation direction from tatoeba
lang_dict = {
    "ceb": {
        "subset": "ceb-eng",
        "source_language": Language.CEBUANO,
        "target_language": Language.ENGLISH,
    },
    "tgl": {
        "subset": "eng-tgl",
        "source_language": Language.ENGLISH,
        "target_language": Language.TAGALOG,
    },
}

FILIPINO_TATOEBA_TASKS = [
    LightevalTaskConfig(
        name=f"tatoeba_{language}",
        prompt_function=get_translation_prompt_function(
            source_language=meta.get("source_language"),
            target_language=meta.get("target_language"),
            adapter=lambda line: {
                "source_text": line["sourceString"],
                "target_text": line["targetString"],
            },
            formulation=CFFormulation(),
        ),
        suite=("filbench",),
        hf_repo="Helsinki-NLP/tatoeba_mt",
        hf_subset=meta.get("subset"),
        metric=[
            Metrics.rougeL,
            Metrics.bleu,
            Metrics.bleurt,
            Metrics.chrf,
            Metrics.ter,
        ],
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        trust_dataset=True,
        generation_size=64,
    )
    for language, meta in lang_dict.items()
]
