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
The News Text References for MT Evaluation of 128 languages (NTREX-128)
dataset contains English news headlines and stories originally
released in WMT19 and translated into 128 target languages including Filipino.
Translations were validated by bilingual annotators who were native speakers
of the respective target language.

Paper link: https://aclanthology.org/2022.sumeval-1.4.pdf
Dataset link: https://huggingface.co/datasets/mteb/NTREX
"""

from langcodes import Language as LangCodeLanguage

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.templates.translation import get_translation_prompt_function
from lighteval.tasks.templates.utils.formulation import CFFormulation
from lighteval.utils.language import Language, iso_639_3_ind_to_iso_639_3_macro


FILIPINO_NTREX_TASK = [
    LightevalTaskConfig(
        name=f"ntrex128_{LangCodeLanguage.get(language).to_alpha3()}",
        prompt_function=get_translation_prompt_function(
            source_language=Language.ENGLISH,
            target_language=iso_639_3_ind_to_iso_639_3_macro[LangCodeLanguage.get(language).to_alpha3()],
            adapter=lambda line: {"source_text": line["eng_Latn"], "target_text": line[language]},
            formulation=CFFormulation(),
        ),
        suite=("filbench",),
        hf_repo="mteb/NTREX",
        hf_subset="default",
        metric=[Metrics.rougeL, Metrics.bleu, Metrics.bleurt, Metrics.chrf, Metrics.ter],
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        few_shots_split=None,
        few_shots_select=None,
        generation_size=64,
        trust_dataset=True,
        version=0,
    )
    for language in ["fil_Latn"]
]
