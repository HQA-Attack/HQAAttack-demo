import csv
import logging
from typing import Dict, Optional

# from allennlp.data.tokenizers.sentence_splitter import SpacySentenceSplitter
import pandas
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, LabelField, TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import SpacyTokenizer
from overrides import overrides
from dne.allennlpx import allenutil

logger = logging.getLogger(__name__)


class SpacyTSVReader(DatasetReader):
    def __init__(self,
                 sent1_col: str,
                 sent2_col: str = None,
                 label_col: str = 'label',
                 max_sequence_length: int = 512,
                 skip_label_indexing: bool = False,
                 lower: bool = True,
                 lazy: bool = False) -> None:
        super().__init__(lazy=lazy)
        self._sent1_col = sent1_col
        self._sent2_col = sent2_col
        self._label_col = label_col
        self._tokenizer = SpacyTokenizer()
        self._max_sequence_length = max_sequence_length
        self._skip_label_indexing = skip_label_indexing
        self._lower = lower
        self._token_indexers = {"tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), "r") as data_file:
            # without the quoting arg, errors will occur with line having quoting characters "/'
            df = pandas.read_csv(data_file, sep='\t', quoting=csv.QUOTE_NONE)
            has_label = self._label_col in df.columns
            for rid in range(0, df.shape[0]):
                sent1 = df.iloc[rid][self._sent1_col]
                if self._lower:
                    sent1 = sent1.lower()

                if self._sent2_col:
                    sent2 = df.iloc[rid][self._sent2_col]
                    if self._lower:
                        sent2 = sent2.lower()
                else:
                    sent2 = None

                if has_label:
                    label = df.iloc[rid][self._label_col]
                    if self._skip_label_indexing:
                        label = int(label)
                else:
                    label = None

                instance = self.text_to_instance(sent1=sent1, sent2=sent2, label=label)
                if instance is not None:
                    yield instance

    @overrides
    def text_to_instance(self,
                         sent1: str,
                         sent2: str = None,
                         label: Optional[str] = None) -> Instance:  # type: ignore
        fields: Dict[str, Field] = {}

        tokens1 = self._tokenizer.tokenize(sent1)[:self._max_sequence_length]
        if sent2:
            tokens2 = self._tokenizer.tokenize(sent2)[:self._max_sequence_length]

        if sent2:
            fields['sent1'] = TextField(tokens1, self._token_indexers)
            fields['sent2'] = TextField(tokens2, self._token_indexers)
        else:
            fields['sent'] = TextField(tokens1, self._token_indexers)

        if label is not None:
            fields['label'] = LabelField(label, skip_indexing=self._skip_label_indexing)
        return Instance(fields)

    def instance_to_text(self, instance: Instance):
        if self._sent2_col is None:
            return {"sent": allenutil.as_sentence(instance['sent'])}
        else:
            return {"sent1": allenutil.as_sentence(instance['sent1']),
                    "sent2": allenutil.as_sentence(instance['sent2'])}
                    