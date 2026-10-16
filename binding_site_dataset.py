"""Dataset used by the ProRB protein/RNA binding-site task.

Expected CSV columns are ``prot``, ``rna``, ``prot_bind`` and ``rna_bind``.
Binding strings contain one character per biological residue/nucleotide:
0 (non-binding), 1 (binding), or 2 (structurally missing).  The loader adds
label 6 at special-token and padding positions.
"""

from pathlib import Path

import pandas as pd
import torch
from esm.sdk.api import ESMProtein
from esm.tokenization import EsmSequenceTokenizer
from multimolecule import RnaTokenizer
from torch.utils.data import Dataset


class BindingSiteDataset(Dataset):
    def __init__(
        self,
        csv_file,
        rnafm_path,
        max_prot_len=1024,
        max_rna_len=512,
    ):
        csv_file = Path(csv_file)
        if not csv_file.is_file():
            raise FileNotFoundError(f"Binding-site CSV not found: {csv_file}")

        self.data = pd.read_csv(
            csv_file,
            dtype={"prot": str, "rna": str, "prot_bind": str, "rna_bind": str},
        )
        required = {"prot", "rna", "prot_bind", "rna_bind"}
        missing = required.difference(self.data.columns)
        if missing:
            raise ValueError(f"Missing CSV columns: {sorted(missing)}")

        self.rna_tokenizer = RnaTokenizer.from_pretrained(rnafm_path)
        self.prot_tokenizer = EsmSequenceTokenizer()
        self.max_prot_len = max_prot_len
        self.max_rna_len = max_rna_len

    def __len__(self):
        return len(self.data)

    @staticmethod
    def _labels(raw, max_length, row_index, name):
        try:
            labels = [int(value) for value in raw]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid {name} at row {row_index}") from exc
        invalid = set(labels).difference({0, 1, 2})
        if invalid:
            raise ValueError(
                f"Unexpected {name} values at row {row_index}: {sorted(invalid)}"
            )
        # Both tokenizers add one token at each end. Q-BioLiP filtering keeps
        # sequences within max_length - 2, but slicing makes the contract safe.
        labels = labels[: max_length - 2]
        labels = [6] + labels + [6]
        labels.extend([6] * (max_length - len(labels)))
        return labels

    @staticmethod
    def _padding_mask(input_ids, pad_token_id):
        return torch.tensor(
            [token_id != pad_token_id for token_id in input_ids], dtype=torch.bool
        )

    @staticmethod
    def _encode(sequence, tokenizer, max_length, protein=False):
        if protein:
            sequence = ESMProtein(sequence=sequence).sequence
        input_ids = tokenizer.encode(
            sequence,
            max_length=max_length,
            padding="max_length",
            truncation=True,
        )
        # EsmSequenceTokenizer does not always pad when called directly.
        input_ids = input_ids[:max_length]
        input_ids.extend([tokenizer.pad_token_id] * (max_length - len(input_ids)))
        return input_ids

    def __getitem__(self, index):
        row = self.data.iloc[index]
        prot_ids = self._encode(
            row["prot"], self.prot_tokenizer, self.max_prot_len, protein=True
        )
        rna_ids = self._encode(
            row["rna"], self.rna_tokenizer, self.max_rna_len, protein=False
        )
        prot_labels = self._labels(
            row["prot_bind"], self.max_prot_len, index, "prot_bind"
        )
        rna_labels = self._labels(
            row["rna_bind"], self.max_rna_len, index, "rna_bind"
        )

        return {
            "prot_input_ids": torch.tensor(prot_ids, dtype=torch.long),
            "rna_input_ids": torch.tensor(rna_ids, dtype=torch.long),
            "prot_bind": torch.tensor(prot_labels, dtype=torch.long),
            "rna_bind": torch.tensor(rna_labels, dtype=torch.long),
            "prot_padding_mask": self._padding_mask(
                prot_ids, self.prot_tokenizer.pad_token_id
            ),
            "rna_padding_mask": self._padding_mask(
                rna_ids, self.rna_tokenizer.pad_token_id
            ),
        }
