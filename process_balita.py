# Copyright (c) 2025 FilBench Team
# %%
import numpy as np
import pandas as pd
from tqdm import tqdm


def process_balita(df: pd.DataFrame, n: int = 4, seed: int = 42) -> pd.DataFrame:
    """
    Args:
        df (pd.DataFrame): Input DataFrame with 'body' and 'title' columns
        n (int): Number of choices to include
        seed (int): Random seed for reproducibility

    Returns:
        pd.DataFrame: New DataFrame with 'text', 'choices', and 'gold_idx' columns
    """
    np.random.seed(seed)
    result_data = []

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        text = row["body"][0]
        correct_title = row["title"]
        other_titles = df[df["title"] != correct_title]["title"].sample(n - 1, random_state=seed).tolist()
        choices = other_titles + [correct_title]
        choices = np.random.permutation(choices).tolist()
        gold_idx = choices.index(correct_title)

        result_data.append({"text": text, "choices": choices, "gold_idx": gold_idx})

    return pd.DataFrame(result_data)


# %%
val = pd.read_json("/home/connermanuel/lighteval/balita-nlp/raw/validation.json")
processed_val = process_balita(val, seed=42)

# %%
