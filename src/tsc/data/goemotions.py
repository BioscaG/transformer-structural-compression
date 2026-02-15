import numpy as np
from datasets import load_dataset, Sequence, Value
from transformers import PreTrainedTokenizerBase

TEXT_COL = "text"
LABEL_COL = "labels"

def load_goemotions(dataset_config: str = "simplified"):
    return load_dataset("go_emotions", dataset_config)

def preprocess_goemotions(ds_split, tokenizer: PreTrainedTokenizerBase, max_length: int):
    label_names = ds_split.features[LABEL_COL].feature.names
    num_labels = len(label_names)

    def preprocess(batch):

        """encoder tokenize the text:
            {
             "input_ids": [[101, ...], [101, ...], ...],
             "attention_mask": [[1,1,1,...], ...] (for marking padded tokens)
            }
        """
        enc = tokenizer(batch[TEXT_COL], truncation=True, max_length=max_length)
        y = np.zeros((len(batch[LABEL_COL]), num_labels), dtype=np.float32)
        for i, labs in enumerate(batch[LABEL_COL]):
            for lab in labs:
                y[i, lab] = 1.0
        enc["labels"] = y
        return enc

    #map of Dataset library with batch for going faster
    ds_tok = ds_split.map(preprocess, batched=True, remove_columns=ds_split.column_names)
    # Ensure float labels for BCEWithLogitsLoss.
    ds_tok = ds_tok.cast_column(LABEL_COL, Sequence(Value("float32")))
    return ds_tok, num_labels