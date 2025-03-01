# FilBench: An Open LLM Leaderboard for Filipino

This repository contains the implementation for FilBench, an Open LLM Leaderboard and Evaluation Suite for Filipino.
It is a fork of HuggingFace's [lighteval](https://github.com/huggingface/lighteval) library, with new Filipino-focused benchmarks implemented under the hood.

## ⌛ Set-up and Installation

First, clone the repository and install all dependencies:

```sh
git clone git@github.com:filbench/lighteval.git
# Create a virtualenv
python3 -m venv venv
pip install -e .[dev]
```

If you're developing FilBench, we encourage installing a pre-commit hook:

```sh
pre-commit install
pre-commit run --all-files
```

## 🔎 Inspecting a task

FilBench contains a suite of evaluation benchmarks from a variety of Filipino datasets.
They are in the following format `filbench|{task_name}|{few_shot}|{truncate_few_shots}`
You can find all tasks in the [examples/tasks/all_filbench_tasks.txt](https://github.com/filbench/lighteval/blob/main/examples/tasks/all_filbench_tasks.txt) file.
For example, let's inspect the `anatomy` subset of [Global-MMLU](https://huggingface.co/datasets/CohereForAI/Global-MMLU):

```sh
python -m lighteval tasks inspect "filbench|global_mmlu_all_tgl_mcf:anatomy|0|0" \
  --num-samples 1 \
  --custom-tasks community_tasks/filbench_evals.py
```

Output:

```python
{ 'choices': [' A', ' B', ' C', ' D'],
  'ctx': '',
  'fewshot_sorting_class': None,
  'gold_index': [0],
  'instruction': '',
  'num_asked_few_shots': -1,
  'num_effective_few_shots': -1,
  'original_query': '',
  'query': 'Tanong: Ang isang sugat na nagdudulot ng compression ng facial '
           'nerve sa stylomastoid foramen ay magdudulot ng ipsilateral\n'
           ' A. Paralysis ng facial muscles.\n'
           ' B. Paralysis ng facial muscles at pagkawala ng panlasa.\n'
           ' C. Paralysis ng facial muscles, pagkawala ng lasa at '
           'lacrimation.\n'
           ' D. Paralysis ng facial muscles, pagkawala ng lasa, lacrimation at '
           'pagbaba ng salivation.\n'
           'Sagot:',
  'specific': None,
  'task_name': 'filbench|global_mmlu_all_tgl_mcf:anatomy',
  'unconditioned_query': 'Sagot:'}
```

> [!TIP]
> Always remember to pass `community_tasks/filbench_evals.py` in the `--custom-tasks` parameter. In addition, running all commands as a module (i.e., using `python -m lighteval` instead of `lighteval`) solves some pathing or weird errors.

You can also check all tasks available in `filbench` (and all of `lighteval`) via this command:

```sh
# Saves all tasks in a file called `all_tasks.txt`
python -m lighteval tasks list --custom-tasks community_tasks/filbench_evals.py > all_tasks.txt
```

## ▶️ Running a task

Please check `lighteval`'s [official documentation on running tasks](https://huggingface.co/docs/lighteval/quicktour).
Nothing much differs except that all of FilBench's tasks are registered in the `filbench` suite.

**(For FilBench developers)** First you need to log-in to HuggingFace or set your `HF_TOKEN`.

```sh
export HF_TOKEN=<your HF token here>
huggingface-cli login  # alternative method for log-in
```

To run on the full evaluation suite, we advise using the following commands:

```sh
# For models in HuggingFace and accessible via vLLM
cat examples/tasks/all_filbench_tasks.txt | xargs -I {} python -m lighteval vllm "pretrained=<MODEL_NAME>" {} --push-to-hub --results-org UD-Filipino --custom-tasks community_tasks/filbench_evals.py

# For models using the OpenAI  API
export OPENAI_API_KEY=<...>
cat examples/tasks/all_filbench_tasks.txt | xargs -I {} python -m lighteval  endpoint openai "<MODEL_NAME>" {} --push-to-hub --results-org UD-Filipino --custom-tasks community_tasks/filbench_evals.py
```

## 🆕 Implementing a new task

Our structure differs quite a bit from the community tasks in `lighteval`.
Specifically, we implement **one task per file** in the `filbench/` directory.
This helps a lot in organization and for multiple people working on different benchmarks at the same time.

1. Implement the task as a new file in the `filbench/` directory. Check if there are similar implementations in the existing tasks in `lighteval`. By default, we follow their implementations to ensure that we're consistent with existing benchmarks. You can check all existing implementations in the `filbench/` directory as reference.
2. Add the task in the `TASK_TABLE` constant in the `community_tasks/filbench_evals.py` file. This file is our main entrypoint for running evaluations.
3. Ensure that nothing is amiss&mdash; inspect the task using `python -m lighteval tasks inspect` to examine a single sample.
4. If everything looks good, add the task string, i.e., `filbench|{task_name}|{few_shot}|{truncate_few_shots}` in the `examples/tasks/all_filbench_tasks.txt` file.

---

<p align="center">
  <br/>
    <img alt="lighteval library logo" src="./assets/lighteval-doc.svg" width="376" height="59" style="max-width: 100%;">
  <br/>
</p>

<p align="center">
    <i>Your go-to toolkit for lightning-fast, flexible LLM evaluation, from Hugging Face's Leaderboard and Evals Team.</i>
</p>

<div align="center">

[![Tests](https://github.com/huggingface/lighteval/actions/workflows/tests.yaml/badge.svg?branch=main)](https://github.com/huggingface/lighteval/actions/workflows/tests.yaml?query=branch%3Amain)
[![Quality](https://github.com/huggingface/lighteval/actions/workflows/quality.yaml/badge.svg?branch=main)](https://github.com/huggingface/lighteval/actions/workflows/quality.yaml?query=branch%3Amain)
[![Python versions](https://img.shields.io/pypi/pyversions/lighteval)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/huggingface/lighteval/blob/main/LICENSE)
[![Version](https://img.shields.io/pypi/v/lighteval)](https://pypi.org/project/lighteval/)

</div>

---

**Documentation**: <a href="https://huggingface.co/docs/lighteval/index" target="_blank">Lighteval's Wiki</a>

---

### Unlock the Power of LLM Evaluation with Lighteval 🚀

**Lighteval** is your all-in-one toolkit for evaluating LLMs across multiple
backends—whether it's
[transformers](https://github.com/huggingface/transformers),
[tgi](https://github.com/huggingface/text-generation-inference),
[vllm](https://github.com/vllm-project/vllm), or
[nanotron](https://github.com/huggingface/nanotron)—with
ease. Dive deep into your model’s performance by saving and exploring detailed,
sample-by-sample results to debug and see how your models stack-up.

Customization at your fingertips: letting you either browse all our existing [tasks](https://huggingface.co/docs/lighteval/available-tasks) and [metrics](https://huggingface.co/docs/lighteval/metric-list) or effortlessly create your own [custom task](https://huggingface.co/docs/lighteval/adding-a-custom-task) and [custom metric](https://huggingface.co/docs/lighteval/adding-a-new-metric), tailored to your needs.

Seamlessly experiment, benchmark, and store your results on the Hugging Face
Hub, S3, or locally.

## 🔑 Key Features

- **Speed**: [Use vllm as backend for fast evals](https://huggingface.co/docs/lighteval/use-vllm-as-backend).
- **Completeness**: [Use the accelerate backend to launch any models hosted on Hugging Face](https://huggingface.co/docs/lighteval/quicktour#accelerate).
- **Seamless Storage**: [Save results in S3 or Hugging Face Datasets](https://huggingface.co/docs/lighteval/saving-and-reading-results).
- **Python API**: [Simple integration with the Python API](https://huggingface.co/docs/lighteval/using-the-python-api).
- **Custom Tasks**: [Easily add custom tasks](https://huggingface.co/docs/lighteval/adding-a-custom-task).
- **Versatility**: Tons of [metrics](https://huggingface.co/docs/lighteval/metric-list) and [tasks](https://huggingface.co/docs/lighteval/available-tasks) ready to go.

## ⚡️ Installation

```bash
pip install lighteval
```

Lighteval allows for many extras when installing, see [here](https://huggingface.co/docs/lighteval/installation) for a complete list.

If you want to push results to the Hugging Face Hub, add your access token as
an environment variable:

```shell
huggingface-cli login
```

## 🚀 Quickstart

Lighteval offers the following entry points for model evaluation:

- `lighteval accelerate` : evaluate models on CPU or one or more GPUs using [🤗
  Accelerate](https://github.com/huggingface/accelerate)
- `lighteval nanotron`: evaluate models in distributed settings using [⚡️
  Nanotron](https://github.com/huggingface/nanotron)
- `lighteval vllm`: evaluate models on one or more GPUs using [🚀
  VLLM](https://github.com/vllm-project/vllm)
- `lighteval endpoint`
  - `inference-endpoint`: evaluate models on one or more GPUs using [🔗
    Inference Endpoint](https://huggingface.co/inference-endpoints/dedicated)
  - `tgi`: evaluate models on one or more GPUs using [🔗 Text Generation Inference](https://huggingface.co/docs/text-generation-inference/en/index)
  - `openai`: evaluate models on one or more GPUs using [🔗 OpenAI API](https://platform.openai.com/)

Here’s a quick command to evaluate using the Accelerate backend:

```shell
lighteval accelerate \
    "pretrained=gpt2" \
    "leaderboard|truthfulqa:mc|0|0"
```

## 🙏 Acknowledgements

Lighteval started as an extension of the fantastic [Eleuther AI
Harness](https://github.com/EleutherAI/lm-evaluation-harness) (which powers the
[Open LLM
Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard))
and draws inspiration from the amazing
[HELM](https://crfm.stanford.edu/helm/latest/) framework.

While evolving Lighteval into its own standalone tool, we are grateful to the
Harness and HELM teams for their pioneering work on LLM evaluations.

## 🌟 Contributions Welcome 💙💚💛💜🧡

Got ideas? Found a bug? Want to add a
[task](https://huggingface.co/docs/lighteval/adding-a-custom-task) or
[metric](https://huggingface.co/docs/lighteval/adding-a-new-metric)?
Contributions are warmly welcomed!

If you're adding a new feature, please open an issue first.

If you open a PR, don't forget to run the styling!

```bash
pip install -e .[dev]
pre-commit install
pre-commit run --all-files
```

## 📜 Citation

```bibtex
@misc{lighteval,
  author = {Fourrier, Clémentine and Habib, Nathan and Kydlíček, Hynek and Wolf, Thomas and Tunstall, Lewis},
  title = {LightEval: A lightweight framework for LLM evaluation},
  year = {2023},
  version = {0.7.0},
  url = {https://github.com/huggingface/lighteval}
}
```
