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
SIB-200 is the largest publicly available topic classification dataset based on Flores-200 covering 205 languages and dialects.
The train/validation/test sets are available for all the 205 languages.

Paper link: https://aclanthology.org/2024.eacl-long.14/
Dataset link: https://huggingface.co/datasets/Davlan/sib200
"""

from collections import OrderedDict
from typing import Callable

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


class CustomFilipinoSIBTask(LightevalTaskConfig):
    def __init__(self, name, lang_code):
        super().__init__(
            name=name,
            hf_subset=f"{lang_code}_Latn",
            prompt_function=fil_sib200_pfn_factory(lang_code),
            hf_repo="Davlan/sib200",
            metric=[Metrics.loglikelihood_acc_norm],
            hf_avail_splits=["test", "validation"],
            evaluation_splits=["validation"],
            few_shots_split=["validation"],
            few_shots_select="sequential",
            suite=["filbench"],
            generation_size=-1,
            stop_sequence=None,
            trust_dataset=True,
            version=0,
        )


def fil_sib200_pfn_factory(lang_code: str) -> Callable:
    if lang_code == "tgl":
        instruction = "Tungkol saan ang sumusunod na pangungusap? Piliin ang tamang sagot:\n\n"
        answer_text = "Sagot:"
    elif lang_code == "ceb":
        instruction = "Mahitungod sa unsa ang mosunod nga teksto? Pilia ang saktong tubag:\n\n"
        answer_text = "Tubag:"
    else:
        raise ValueError(f"Unknown lang_code {lang_code}")

    translation = {
        "tgl": {
            "geography": "heograpiya",
            "science/technology": "agham/teknolohiya",
            "entertainment": "libangan",
            "travel": "pagbiyahe",
            "sports": "isports",
            "health": "kalusugan",
            "politics": "politika",
        },
        "ceb": {
            "geography": "heograpiya",
            "science/technology": "agham/teknolohiya",
            "entertainment": "kalingawan",
            "travel": "pagbiyahe",
            "sports": "isports",
            "health": "panglawas",
            "politics": "politika",
        },
    }

    eng_to_fil = translation[lang_code]

    def fil_sib200_pfn(line, task_name: str = None) -> Doc:
        choices: dict[str, str] = OrderedDict(
            {
                "A": "geography",
                "B": "science/technology",
                "C": "entertainment",
                "D": "travel",
                "E": "sports",
                "F": "health",
                "G": "politics",
            }
        )

        answer_index = list(choices.values()).index(line.get("label"))
        query = f"{instruction}{line['text']}\n"
        query += "".join([f"{key}. {eng_to_fil[choice]}\n" for key, choice in choices.items()])
        query += answer_text
        return Doc(
            task_name=task_name,
            query=query,
            choices=list(choices.keys()),
            gold_index=answer_index,
            instruction=instruction,
        )

    return fil_sib200_pfn


FILIPINO_SIB_TASKS = [
    CustomFilipinoSIBTask(name=f"filipino_sib200:{lang_code}", lang_code=lang_code) for lang_code in ("tgl", "ceb")
]
