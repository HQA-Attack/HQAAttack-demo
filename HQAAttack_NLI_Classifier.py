import pdb
import sys
import pickle
import argparse
import os
from pathlib import Path
import numpy as np
from scipy import spatial

np.random.seed(1234)
from scipy.special import softmax
import fnmatch
import criteria
import string
import pickle
import random
import json
 
random.seed(0)
import csv

from InferSent.models import NLINet, InferSent
from esim.model import ESIM
from esim.data import Preprocessor
from esim.utils import correct_predictions
from collections import defaultdict
import tensorflow.compat.v1 as tf

# To make tf 2.0 compatible with tf1.0 code, we disable the tf2.0 functionalities
tf.compat.v1.disable_eager_execution()
import tensorflow_hub as hub
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SequentialSampler, TensorDataset
from BERT.tokenization import BertTokenizer
from BERT.modeling import BertForSequenceClassification, BertConfig


class NLI_infer_InferSent(nn.Module):
    def __init__(self,
                 pretrained_file,
                 embedding_path,
                 data,
                 batch_size=32):
        super(NLI_infer_InferSent, self).__init__()

        #         self.device = torch.device("cuda:{}".format(local_rank) if local_rank > -1 else "cpu")
        # torch.cuda.set_device(local_rank)

        # Retrieving model parameters from checkpoint.
        config_nli_model = {
            'word_emb_dim': 300,
            'enc_lstm_dim': 2048,
            'n_enc_layers': 1,
            'dpout_model': 0.,
            'dpout_fc': 0.,
            'fc_dim': 512,
            'bsize': batch_size,
            'n_classes': 3,
            'pool_type': 'max',
            'nonlinear_fc': 0,
            'encoder_type': 'InferSent',
            'use_cuda': True,
            'use_target': False,
            'version': 1,
        }
        params_model = {'bsize': 64, 'word_emb_dim': 200, 'enc_lstm_dim': 2048,
                        'pool_type': 'max', 'dpout_model': 0.0, 'version': 1}

        print("\t* Building model...")
        self.model = NLINet(config_nli_model).cuda()
        print("Reloading pretrained parameters...")
        self.model.load_state_dict(torch.load(os.path.join("savedir/", "model.pickle")))

        # construct dataset loader
        print('Building vocab and embeddings...')
        self.dataset = NLIDataset_InferSent(embedding_path, data=data, batch_size=batch_size)

    def text_pred(self, text_data):
        # Switch the model to eval mode.
        self.model.eval()

        # transform text data into indices and create batches
        data_batches = self.dataset.transform_text(text_data)

        # Deactivate autograd for evaluation.
        probs_all = []
        with torch.no_grad():
            for batch in data_batches:
                # Move input and output data to the GPU if one is used.
                (s1_batch, s1_len), (s2_batch, s2_len) = batch
                s1_batch, s2_batch = s1_batch.cuda(), s2_batch.cuda()
                logits = self.model((s1_batch, s1_len), (s2_batch, s2_len))
                probs = nn.functional.softmax(logits, dim=-1)
                probs_all.append(probs)

        return torch.cat(probs_all, dim=0)


class NLI_infer_ESIM(nn.Module):
    def __init__(self,
                 pretrained_file,
                 worddict_path,
                 local_rank=-1,
                 batch_size=32):
        super(NLI_infer_ESIM, self).__init__()

        self.batch_size = batch_size
        self.device = torch.device("cuda:{}".format(local_rank) if local_rank > -1 else "cuda")
        checkpoint = torch.load(pretrained_file)
        # Retrieving model parameters from checkpoint.
        vocab_size = checkpoint['model']['_word_embedding.weight'].size(0)
        embedding_dim = checkpoint['model']['_word_embedding.weight'].size(1)
        hidden_size = checkpoint['model']['_projection.0.weight'].size(0)
        num_classes = checkpoint['model']['_classification.4.weight'].size(0)

        print("\t* Building model...")
        self.model = ESIM(vocab_size,
                          embedding_dim,
                          hidden_size,
                          num_classes=num_classes,
                          device=self.device).to(self.device)

        self.model.load_state_dict(checkpoint['model'])

        # construct dataset loader
        self.dataset = NLIDataset_ESIM(worddict_path)

    def text_pred(self, text_data):
        # Switch the model to eval mode.
        self.model.eval()
        device = self.device

        # transform text data into indices and create batches
        self.dataset.transform_text(text_data)
        dataloader = DataLoader(self.dataset, shuffle=False, batch_size=self.batch_size)

        # Deactivate autograd for evaluation.
        probs_all = []
        with torch.no_grad():
            for batch in dataloader:
                # Move input and output data to the GPU if one is used.
                premises = batch['premise'].to(device)
                premises_lengths = batch['premise_length'].to(device)
                hypotheses = batch['hypothesis'].to(device)
                hypotheses_lengths = batch['hypothesis_length'].to(device)

                _, probs = self.model(premises,
                                      premises_lengths,
                                      hypotheses,
                                      hypotheses_lengths)
                probs_all.append(probs)

        return torch.cat(probs_all, dim=0)


class NLI_infer_BERT(nn.Module):
    def __init__(self,
                 pretrained_dir,
                 max_seq_length=128,
                 batch_size=32):
        super(NLI_infer_BERT, self).__init__()
        self.model = BertForSequenceClassification.from_pretrained(pretrained_dir, num_labels=3).cuda()

        # construct dataset loader
        self.dataset = NLIDataset_BERT(pretrained_dir, max_seq_length=max_seq_length, batch_size=batch_size)

    def text_pred(self, text_data):
        # Switch the model to eval mode.
        self.model.eval()

        # transform text data into indices and create batches
        dataloader = self.dataset.transform_text(text_data)

        probs_all = []
        #         for input_ids, input_mask, segment_ids in tqdm(dataloader, desc="Evaluating"):
        for input_ids, input_mask, segment_ids in dataloader:
            input_ids = input_ids.cuda()
            input_mask = input_mask.cuda()
            segment_ids = segment_ids.cuda()

            with torch.no_grad():
                logits = self.model(input_ids, segment_ids, input_mask)
                probs = nn.functional.softmax(logits, dim=-1)
                probs_all.append(probs)

        return torch.cat(probs_all, dim=0)


tf.compat.v1.disable_eager_execution()
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024 * 1.2)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)


class USE(object):
    def __init__(self, cache_path):
        super(USE, self).__init__()
        os.environ['TFHUB_CACHE_DIR'] = cache_path
        module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"
        self.embed = hub.Module(module_url)

        self.sess = tf.Session()
        self.build_graph()
        self.sess.run([tf.global_variables_initializer(), tf.tables_initializer()])

    def build_graph(self):
        self.sts_input1 = tf.placeholder(tf.string, shape=(None))
        self.sts_input2 = tf.placeholder(tf.string, shape=(None))

        sts_encode1 = tf.nn.l2_normalize(self.embed(self.sts_input1), axis=1)
        sts_encode2 = tf.nn.l2_normalize(self.embed(self.sts_input2), axis=1)
        self.cosine_similarities = tf.reduce_sum(tf.multiply(sts_encode1, sts_encode2), axis=1)
        clip_cosine_similarities = tf.clip_by_value(self.cosine_similarities, -1.0, 1.0)
        self.sim_scores = 1.0 - tf.acos(clip_cosine_similarities)

    def semantic_sim(self, sents1, sents2):
        scores = self.sess.run(
            [self.sim_scores],
            feed_dict={
                self.sts_input1: sents1,
                self.sts_input2: sents2,
            })
        return scores


def pick_most_similar_words_batch(src_words, sim_mat, idx2word, ret_count=10, threshold=0.):
    """
    embeddings is a matrix with (d, vocab_size)
    """
    sim_order = np.argsort(-sim_mat[src_words, :])[:, 1:1 + ret_count]
    sim_words, sim_values = [], []
    for idx, src_word in enumerate(src_words):
        sim_value = sim_mat[src_word][sim_order[idx]]
        mask = sim_value >= threshold
        sim_word, sim_value = sim_order[idx][mask], sim_value[mask]
        sim_word = [idx2word[id] for id in sim_word]
        sim_words.append(sim_word)
        sim_values.append(sim_value)
    return sim_words, sim_values


def read_data(filepath, data_size, target_model='infersent', lowercase=False, ignore_punctuation=False, stopwords=[]):
    """
    Read the premises, hypotheses and labels from some NLI dataset's
    file and return them in a dictionary. The file should be in the same
    form as SNLI's .txt files.

    Args:
        filepath: The path to a file containing some premises, hypotheses
            and labels that must be read. The file should be formatted in
            the same way as the SNLI (and MultiNLI) dataset.

    Returns:
        A dictionary containing three lists, one for the premises, one for
        the hypotheses, and one for the labels in the input data.
    """
    if target_model == 'bert':
        labeldict = {"contradiction": 0,
                     "entailment": 1,
                     "neutral": 2}
    else:
        labeldict = {"entailment": 0,
                     "neutral": 1,
                     "contradiction": 2}
    with open(filepath, 'r', encoding='utf8') as input_data:
        premises, hypotheses, labels = [], [], []

        # Translation tables to remove punctuation from strings.
        punct_table = str.maketrans({key: ' '
                                     for key in string.punctuation})

        for idx, line in enumerate(input_data):
            if idx >= data_size:
                break

            line = line.strip().split('\t')

            # Ignore sentences that have no gold label.
            if line[0] == '-':
                continue

            premise = line[1]
            hypothesis = line[2]

            if lowercase:
                premise = premise.lower()
                hypothesis = hypothesis.lower()

            if ignore_punctuation:
                premise = premise.translate(punct_table)
                hypothesis = hypothesis.translate(punct_table)

            # Each premise and hypothesis is split into a list of words.
            premises.append([w for w in premise.rstrip().split()
                             if w not in stopwords])
            hypotheses.append([w for w in hypothesis.rstrip().split()
                               if w not in stopwords])
            labels.append(labeldict[line[0]])

        return {"premises": premises,
                "hypotheses": hypotheses,
                "labels": labels}


class NLIDataset_ESIM(Dataset):
    """
    Dataset class for Natural Language Inference datasets.

    The class can be used to read preprocessed datasets where the premises,
    hypotheses and labels have been transformed to unique integer indices
    (this can be done with the 'preprocess_data' script in the 'scripts'
    folder of this repository).
    """

    def __init__(self,
                 worddict_path,
                 padding_idx=0,
                 bos="_BOS_",
                 eos="_EOS_"):
        """
        Args:
            data: A dictionary containing the preprocessed premises,
                hypotheses and labels of some dataset.
            padding_idx: An integer indicating the index being used for the
                padding token in the preprocessed data. Defaults to 0.
            max_premise_length: An integer indicating the maximum length
                accepted for the sequences in the premises. If set to None,
                the length of the longest premise in 'data' is used.
                Defaults to None.
            max_hypothesis_length: An integer indicating the maximum length
                accepted for the sequences in the hypotheses. If set to None,
                the length of the longest hypothesis in 'data' is used.
                Defaults to None.
        """
        self.bos = bos
        self.eos = eos
        self.padding_idx = padding_idx

        # build word dict
        with open(worddict_path, 'rb') as pkl:
            self.worddict = pickle.load(pkl)

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, index):
        return {
            "premise": self.data["premises"][index],
            "premise_length": min(self.premises_lengths[index],
                                  self.max_premise_length),
            "hypothesis": self.data["hypotheses"][index],
            "hypothesis_length": min(self.hypotheses_lengths[index],
                                     self.max_hypothesis_length)
        }

    def words_to_indices(self, sentence):
        """
        Transform the words in a sentence to their corresponding integer
        indices.

        Args:
            sentence: A list of words that must be transformed to indices.

        Returns:
            A list of indices.
        """
        indices = []
        # Include the beggining of sentence token at the start of the sentence
        # if one is defined.
        if self.bos:
            indices.append(self.worddict["_BOS_"])

        for word in sentence:
            if word in self.worddict:
                index = self.worddict[word]
            else:
                # Words absent from 'worddict' are treated as a special
                # out-of-vocabulary word (OOV).
                index = self.worddict['_OOV_']
            indices.append(index)
        # Add the end of sentence token at the end of the sentence if one
        # is defined.
        if self.eos:
            indices.append(self.worddict["_EOS_"])

        return indices

    def transform_to_indices(self, data):
        """
        Transform the words in the premises and hypotheses of a dataset, as
        well as their associated labels, to integer indices.

        Args:
            data: A dictionary containing lists of premises, hypotheses
                and labels, in the format returned by the 'read_data'
                method of the Preprocessor class.

        Returns:
            A dictionary containing the transformed premises, hypotheses and
            labels.
        """
        transformed_data = {"premises": [],
                            "hypotheses": []}

        for i, premise in enumerate(data['premises']):
            # Ignore sentences that have a label for which no index was
            # defined in 'labeldict'.

            indices = self.words_to_indices(premise)
            transformed_data["premises"].append(indices)

            indices = self.words_to_indices(data["hypotheses"][i])
            transformed_data["hypotheses"].append(indices)

        return transformed_data

    def transform_text(self, data):
        #         # standardize data format
        #         data = defaultdict(list)
        #         for hypothesis in hypotheses:
        #             data['premises'].append(premise)
        #             data['hypotheses'].append(hypothesis)

        # transform data into indices
        data = self.transform_to_indices(data)

        self.premises_lengths = [len(seq) for seq in data["premises"]]
        self.max_premise_length = max(self.premises_lengths)

        self.hypotheses_lengths = [len(seq) for seq in data["hypotheses"]]
        self.max_hypothesis_length = max(self.hypotheses_lengths)

        self.num_sequences = len(data["premises"])

        self.data = {
            "premises": torch.ones((self.num_sequences,
                                    self.max_premise_length),
                                   dtype=torch.long) * self.padding_idx,
            "hypotheses": torch.ones((self.num_sequences,
                                      self.max_hypothesis_length),
                                     dtype=torch.long) * self.padding_idx}

        for i, premise in enumerate(data["premises"]):
            end = min(len(premise), self.max_premise_length)
            self.data["premises"][i][:end] = torch.tensor(premise[:end])

            hypothesis = data["hypotheses"][i]
            end = min(len(hypothesis), self.max_hypothesis_length)
            self.data["hypotheses"][i][:end] = torch.tensor(hypothesis[:end])


class NLIDataset_InferSent(Dataset):
    """
    Dataset class for Natural Language Inference datasets.

    The class can be used to read preprocessed datasets where the premises,
    hypotheses and labels have been transformed to unique integer indices
    (this can be done with the 'preprocess_data' script in the 'scripts'
    folder of this repository).
    """

    def __init__(self,
                 embedding_path,
                 data,
                 word_emb_dim=300,
                 batch_size=32,
                 bos="<s>",
                 eos="</s>"):
        """
        Args:
            data: A dictionary containing the preprocessed premises,
                hypotheses and labels of some dataset.
            padding_idx: An integer indicating the index being used for the
                padding token in the preprocessed data. Defaults to 0.
            max_premise_length: An integer indicating the maximum length
                accepted for the sequences in the premises. If set to None,
                the length of the longest premise in 'data' is used.
                Defaults to None.
            max_hypothesis_length: An integer indicating the maximum length
                accepted for the sequences in the hypotheses. If set to None,
                the length of the longest hypothesis in 'data' is used.
                Defaults to None.
        """
        self.bos = bos
        self.eos = eos
        self.word_emb_dim = word_emb_dim
        self.batch_size = batch_size

        # build word dict
        self.word_vec = self.build_vocab(data['premises'] + data['hypotheses'], embedding_path)

    def build_vocab(self, sentences, embedding_path):
        word_dict = self.get_word_dict(sentences)
        word_vec = self.get_embedding(word_dict, embedding_path)
        print('Vocab size : {0}'.format(len(word_vec)))
        return word_vec

    def get_word_dict(self, sentences):
        # create vocab of words
        word_dict = {}
        for sent in sentences:
            for word in sent:
                if word not in word_dict:
                    word_dict[word] = ''
        word_dict['<s>'] = ''
        word_dict['</s>'] = ''
        word_dict['<oov>'] = ''
        return word_dict

    def get_embedding(self, word_dict, embedding_path):
        # create word_vec with glove vectors
        word_vec = {}
        word_vec['<oov>'] = np.random.normal(size=(self.word_emb_dim))
        with open(embedding_path) as f:
            for line in f:
                word, vec = line.split(' ', 1)
                if word in word_dict:
                    word_vec[word] = np.array(list(map(float, vec.split())))
        print('Found {0}(/{1}) words with embedding vectors'.format(
            len(word_vec), len(word_dict)))
        return word_vec

    def get_batch(self, batch, word_vec, emb_dim=300):
        # sent in batch in decreasing order of lengths (bsize, max_len, word_dim)
        lengths = np.array([len(x) for x in batch])
        max_len = np.max(lengths)
        #         print(max_len)
        embed = np.zeros((max_len, len(batch), emb_dim))

        for i in range(len(batch)):
            for j in range(len(batch[i])):
                if batch[i][j] in word_vec:
                    embed[j, i, :] = word_vec[batch[i][j]]
                else:
                    embed[j, i, :] = word_vec['<oov>']
        #                     embed[j, i, :] = np.random.normal(size=(emb_dim))

        return torch.from_numpy(embed).float(), lengths

    def transform_text(self, data):
        # transform data into seq of embeddings
        premises = data['premises']
        hypotheses = data['hypotheses']

        # add bos and eos
        premises = [['<s>'] + premise + ['</s>'] for premise in premises]
        hypotheses = [['<s>'] + hypothese + ['</s>'] for hypothese in hypotheses]

        batches = []
        for stidx in range(0, len(premises), self.batch_size):
            # prepare batch
            s1_batch, s1_len = self.get_batch(premises[stidx:stidx + self.batch_size],
                                              self.word_vec, self.word_emb_dim)
            s2_batch, s2_len = self.get_batch(hypotheses[stidx:stidx + self.batch_size],
                                              self.word_vec, self.word_emb_dim)
            batches.append(((s1_batch, s1_len), (s2_batch, s2_len)))

        return batches


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids


class NLIDataset_BERT(Dataset):
    """
    Dataset class for Natural Language Inference datasets.

    The class can be used to read preprocessed datasets where the premises,
    hypotheses and labels have been transformed to unique integer indices
    (this can be done with the 'preprocess_data' script in the 'scripts'
    folder of this repository).
    """

    def __init__(self,
                 pretrained_dir,
                 max_seq_length=128,
                 batch_size=32):
        """
        Args:
            data: A dictionary containing the preprocessed premises,
                hypotheses and labels of some dataset.
            padding_idx: An integer indicating the index being used for the
                padding token in the preprocessed data. Defaults to 0.
            max_premise_length: An integer indicating the maximum length
                accepted for the sequences in the premises. If set to None,
                the length of the longest premise in 'data' is used.
                Defaults to None.
            max_hypothesis_length: An integer indicating the maximum length
                accepted for the sequences in the hypotheses. If set to None,
                the length of the longest hypothesis in 'data' is used.
                Defaults to None.
        """
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_dir, do_lower_case=True)
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size

    def _truncate_seq_pair(self, tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def convert_examples_to_features(self, examples, max_seq_length, tokenizer):
        """Loads a data file into a list of `InputBatch`s."""

        features = []
        for (ex_index, (text_a, text_b)) in enumerate(examples):
            tokens_a = tokenizer.tokenize(' '.join(text_a))

            tokens_b = None
            if text_b:
                tokens_b = tokenizer.tokenize(' '.join(text_b))
                # Modifies `tokens_a` and `tokens_b` in place so that the total
                # length is less than the specified length.
                # Account for [CLS], [SEP], [SEP] with "- 3"
                self._truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
            else:
                # Account for [CLS] and [SEP] with "- 2"
                if len(tokens_a) > max_seq_length - 2:
                    tokens_a = tokens_a[:(max_seq_length - 2)]

            tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
            segment_ids = [0] * len(tokens)

            if tokens_b:
                tokens += tokens_b + ["[SEP]"]
                segment_ids += [1] * (len(tokens_b) + 1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids))
        return features

    def transform_text(self, data):
        # transform data into seq of embeddings
        eval_features = self.convert_examples_to_features(list(zip(data['premises'], data['hypotheses'])),
                                                          self.max_seq_length, self.tokenizer)

        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)

        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=self.batch_size)

        return eval_dataloader


# It calculates semantic similarity between two text inputs.
# text_ls (list): First text input either original text input or previous text.
# new_texts (list): Updated text inputs.
# idx (int): Index of the word that has been changed.
# sim_score_window (int): The number of words to consider around idx. If idx = -1 consider the whole text.
def calc_sim(text_ls, new_texts, idx, sim_score_window, sim_predictor):
    len_text = len(text_ls)
    half_sim_score_window = (sim_score_window - 1) // 2

    # Compute the starting and ending indices of the window.
    if idx >= half_sim_score_window and len_text - idx - 1 >= half_sim_score_window:
        text_range_min = idx - half_sim_score_window
        text_range_max = idx + half_sim_score_window + 1
    elif idx < half_sim_score_window and len_text - idx - 1 >= half_sim_score_window:
        text_range_min = 0
        text_range_max = sim_score_window
    elif idx >= half_sim_score_window and len_text - idx - 1 < half_sim_score_window:
        text_range_min = len_text - sim_score_window
        text_range_max = len_text
    else:
        text_range_min = 0
        text_range_max = len_text

    if text_range_min < 0:
        text_range_min = 0
    if text_range_max > len_text:
        text_range_max = len_text

    if idx == -1:
        text_rang_min = 0
        text_range_max = len_text

    semantic_sims = \
        sim_predictor.semantic_sim([' '.join(text_ls[text_range_min:text_range_max])],
                                   list(map(lambda x: ' '.join(x[text_range_min:text_range_max]), new_texts)))[0]

    return semantic_sims


# Returns the hard label prediction of the target model.
# new_text (list): Text to be fed to target model.
# predictor: Target Model.
# orig_label (int): Original label.
# batch_size (int): Batch size.
def get_attack_result(hypotheses, premise, predictor, orig_label, batch_size):
    new_probs = predictor({'premises': [premise] * len(hypotheses), 'hypotheses': hypotheses})
    pr = (orig_label != torch.argmax(new_probs, dim=-1)).data.cpu().numpy()
    return pr






def random_attack(fuzz_val, top_k_words, qrs, sample_index, hypotheses, premise, true_label,
                  predictor, stop_words_set, word2idx, idx2word, cos_sim, sim_predictor=None,
                  import_score_threshold=-1., sim_score_threshold=0.5, sim_score_window=15, synonym_num=50,
                  batch_size=32):
    # first check the prediction of the original text

    orig_probs = predictor(
        {'premises': [premise], 'hypotheses': [hypotheses]}).squeeze()  # predictor(premise,hypothese).squeeze()
    orig_label = torch.argmax(orig_probs)

    print(orig_label)
    orig_prob = orig_probs.max()

    if true_label != orig_label:
        return 0, 0, 0, False
    else:
        text_ls = hypotheses[:]
        pos_ls = criteria.get_pos(text_ls)
        len_text = len(text_ls)
        if len_text < sim_score_window:
            sim_score_threshold = 0.1  # shut down the similarity thresholding function
        half_sim_score_window = (sim_score_window - 1) // 2
        num_queries = 1
        rank = {}
        # get the pos and verb tense info
        words_perturb = []
        pos_ls = criteria.get_pos(text_ls)
        pos_pref = ["ADJ", "ADV", "VERB", "NOUN"]
        for pos in pos_pref:
            for i in range(len(pos_ls)):
                if pos_ls[i] == pos and len(text_ls[i]) > 2:
                    words_perturb.append((i, text_ls[i]))

        random.shuffle(words_perturb)

        # find synonyms and make a dict of synonyms of each word.
        words_perturb = words_perturb[:top_k_words]
        words_perturb_idx = [word2idx[word] for idx, word in words_perturb if word in word2idx]
        synonym_words, synonym_values = [], []
        for idx in words_perturb_idx:
            res = list(zip(*(cos_sim[idx])))
            temp = []
            for ii in res[1]:
                temp.append(idx2word[ii])
            synonym_words.append(temp)
            temp = []
            for ii in res[0]:
                temp.append(ii)
            synonym_values.append(temp)
        synonyms_all = []
        synonyms_dict = defaultdict(list)
        for idx, word in words_perturb:
            if word in word2idx:
                synonyms = synonym_words.pop(0)
                if synonyms:
                    synonyms_all.append((idx, synonyms))
                    synonyms_dict[word] = synonyms

        # STEP 1: Random initialisation.
        qrs = 0
        num_changed = 0
        flag = 0
        th = 0

        while qrs < len(text_ls):
            random_text = text_ls[:]
            for i in range(len(synonyms_all)):
                idx = synonyms_all[i][0]
                syn = synonyms_all[i][1]
                random_text[idx] = random.choice(syn)
                if i >= th:
                    break
            pr = get_attack_result([random_text], premise, predictor, orig_label, batch_size)
            qrs += 1
            th += 1
            if th > len_text:
                break
            if np.sum(pr) > 0:
                flag = 1
                break
        old_qrs = qrs

        while qrs < old_qrs + 2500 and flag == 0:
            random_text = text_ls[:]
            for j in range(len(synonyms_all)):
                idx = synonyms_all[j][0]
                syn = synonyms_all[j][1]
                random_text[idx] = random.choice(syn)
                if j >= len_text:
                    break
            pr = get_attack_result([random_text], premise, predictor, orig_label, batch_size)
            qrs += 1
            if np.sum(pr) > 0:
                flag = 1
                break

        if flag == 1:
            return random_text, qrs, orig_label, True
        else:
            return random_text, qrs,orig_label,False

def get_pert_rate(text_str,ades_str):
    text_ls = text_str.split()
    ades_ls = ades_str.split()
    changed =0
    for i,j in zip(text_ls,ades_ls):
        if i!=j:
            changed += 1
    return changed / len(text_ls)

l2s = lambda l: " ".join(l)
def cos_sim_compute(x,y):
    x = torch.FloatTensor(x)
    y = torch.FloatTensor(y)
    return float(torch.cosine_similarity(x,y,dim=0).numpy())

def fn(sim_ls):
    abs_sum = np.sum([np.abs(i) for i in sim_ls])
    for j in range(len(sim_ls)):
        sim_ls[j] = sim_ls[j]/abs_sum

def softmax(x):
    x = np.array(x)
    y = np.exp(x - np.max(x))
    f_x = y / np.sum(np.exp(x))
    return f_x


def attack(fuzz_val, top_k_words,optim_step, qrs, sample_index, hypotheses,random_text_, premise, true_label,orig_label,
           predictor, stop_words_set, word2idx, idx2word, cos_sim, sim_predictor=None,n_sample = 5,
           import_score_threshold=-1., sim_score_threshold=0.5, sim_score_window=15, synonym_num=50,
           batch_size=32, embed_func='',k_threshold=0,k_sample=5,qrs_limits=1000,qrs_stopp=[100,200,300,400,500,600,700,800,900,1000]):

    orig_probs = predictor(
        {'premises': [premise], 'hypotheses': [hypotheses]}).squeeze()
    orig_label = torch.argmax(orig_probs)
    orig_prob = orig_probs.max()

    if true_label != orig_label:
        return '', 0, 0, orig_label, orig_label, 0, 0, 0
    else:
        word_idx_dict = {}
        with open(embed_func, 'r') as ifile:
            for index, line in enumerate(ifile):
                word = line.strip().split()[0]
                word_idx_dict[word] = index

        embed_file = open(embed_func)
        embed_content = embed_file.readlines()


        text_ls = hypotheses[:]
        pos_ls = criteria.get_pos(text_ls)
        len_text = len(text_ls)
        if len_text < sim_score_window:
            sim_score_threshold = 0.1  # shut down the similarity thresholding function
        half_sim_score_window = (sim_score_window - 1) // 2
        num_queries = 1
        rank = {}
        # get the pos and verb tense info
        words_perturb = []
        pos_ls = criteria.get_pos(text_ls)
        pos_pref = ["ADJ", "ADV", "VERB", "NOUN"]
        for pos in pos_pref:
            for i in range(len(pos_ls)):
                if pos_ls[i] == pos and len(text_ls[i]) > 2:
                    words_perturb.append((i, text_ls[i]))

        random.shuffle(words_perturb)

        # find synonyms and make a dict of synonyms of each word.
        words_perturb = words_perturb[:top_k_words]
        words_perturb_idx = [word2idx[word] for idx, word in words_perturb if word in word2idx]
        synonym_words, synonym_values = [], []
        for idx in words_perturb_idx:
            res = list(zip(*(cos_sim[idx])))
            temp = []
            for ii in res[1]:
                temp.append(idx2word[ii])
            synonym_words.append(temp)
            temp = []
            for ii in res[0]:
                temp.append(ii)
            synonym_values.append(temp)
        synonyms_all = []
        synonyms_dict = defaultdict(list)
        for idx, word in words_perturb:
            if word in word2idx:
                synonyms = synonym_words.pop(0)
                if synonyms:
                    synonyms_all.append((idx, synonyms))
                    synonyms_dict[word] = synonyms

        # STEP 1: Random initialisation.
        qrs = 0
        num_changed = 0
        flag = 0
        th = 0

        while qrs < len(text_ls):
            random_text = text_ls[:]
            for i in range(len(synonyms_all)):
                idx = synonyms_all[i][0]
                syn = synonyms_all[i][1]
                random_text[idx] = random.choice(syn)
                if i >= th:
                    break
            pr = get_attack_result([random_text], premise, predictor, orig_label, batch_size)
            qrs += 1
            th += 1
            if th > len_text:
                break
            if np.sum(pr) > 0:
                flag = 1
                break
        old_qrs = qrs

        while qrs < old_qrs + 2500 and flag == 0:
            random_text = text_ls[:]
            for j in range(len(synonyms_all)):
                idx = synonyms_all[j][0]
                syn = synonyms_all[j][1]
                random_text[idx] = random.choice(syn)
                if j >= len_text:
                    break
            pr = get_attack_result([random_text], premise, predictor, orig_label, batch_size)
            qrs += 1
            if np.sum(pr) > 0:
                flag = 1
                break

        if flag == 1:
            words_perturb_idx = []
            words_perturb_embed = []
            words_perturb_doc_idx = []
            for idx, word in words_perturb:
                if word in word_idx_dict:
                    words_perturb_doc_idx.append(idx)
                    words_perturb_idx.append(word2idx[word])
                    words_perturb_embed.append(
                        [float(num) for num in embed_content[word_idx_dict[word]].strip().split()[1:]])

            words_perturb_embed_matrix = np.asarray(words_perturb_embed)


            changed = 0
            for i in range(len(text_ls)):
                if text_ls[i] != random_text[i]:
                    changed += 1
            print(changed)

            changed_indices = []
            num_changed = 0
            for i in range(len(text_ls)):
                if text_ls[i] != random_text[i]:
                    changed_indices.append(i)
                    num_changed += 1
            random_sim = calc_sim(text_ls, [random_text], -1, sim_score_window, sim_predictor)[0]
            x_t = random_text[:]
            if num_changed == 1:
                change_idx = 0
                for i in range(len(text_ls)):
                    if text_ls[i] != x_t[i]:
                        change_idx = i
                        break
                idx = word2idx[text_ls[change_idx]]
                res = list(zip(*(cos_sim[idx])))
                best_attack = random_text
                best_sim = random_sim
                for widx in res[1]:
                    w = idx2word[widx]
                    x_t[change_idx] = w
                    pr = get_attack_result([x_t],premise, predictor, orig_label, batch_size)
                    sim = calc_sim(text_ls, [x_t], -1, sim_score_window, sim_predictor)[0]
                    qrs += 1
                    if np.sum(pr) > 0:
                        if sim>=best_sim:
                            best_attack = x_t[:]
                            best_sim = calc_sim(text_ls, [x_t], -1, sim_score_window, sim_predictor)[0]
                return ' '.join(best_attack), 1, 1, \
                    orig_label, torch.argmax(
                    predictor({'premises': [premise], 'hypotheses': [best_attack]})), qrs, best_sim, random_sim

            best_attack = random_text[:]
            best_sim = calc_sim(text_ls, [best_attack], -1, sim_score_window, sim_predictor)[0]

            x_tilde = random_text[:]

            stack = [random_text[:]]
            stack_str = []
            stack_over = []
            way_back_num = 3
            wbcount = 0
            for t in range(100):

                x_t = x_tilde[:]

                x_t_str = " ".join(x_t)
                if x_t_str not in stack_over and x_t_str not in stack_str:
                    pr = get_attack_result([x_t], premise, predictor, orig_label, batch_size)
                    if np.sum(pr) > 0:
                        stack.append(x_t[:])
                        stack_str.append(" ".join(x_t))
                num_changed = 0
                for i, j in zip(x_t, text_ls):
                    if i != j:
                        num_changed += 1

                if wbcount > way_back_num:
                    if len(stack) > 5:
                        popint = random.randint(3, 5)
                        for _ in range(popint):
                            x_t = stack.pop()
                            stack_str.pop()
                        stack_over.append(" ".join(x_t))
                    else:
                        x_t = random_text[:]
                    wbcount = 0


                while True:
                    choices = []

                    for i in range(len(text_ls)):
                        if x_t[i] != text_ls[i]:
                            new_text = x_t[:]
                            new_text[i] = text_ls[i]
                            semantic_sims = calc_sim(text_ls, [new_text], -1, sim_score_window, sim_predictor)
                            choices.append((i, semantic_sims[0]))


                    flag = True
                    if len(choices) > 0:
                        choices.sort(key=lambda x: x[1])
                        choices.reverse()
                        
                        for i in range(len(choices)):
                            new_text = x_t[:]
                            new_text[choices[i][0]] = text_ls[choices[i][0]]
                            pr = get_attack_result([new_text], premise, predictor, orig_label, batch_size)
                            qrs += 1
                            if pr[0] != 0:
                                flag = False
                                x_t[choices[i][0]] = text_ls[choices[i][0]]
                                break
                    if flag:
                        break

                    if len(choices) == 0:
                        break

                num_changed = 0
                for i in range(len(text_ls)):
                    if text_ls[i] != x_t[i]:
                        num_changed += 1
                x_t_sim = calc_sim(text_ls, [x_t], -1, sim_score_window, sim_predictor)[0]

                if np.sum(get_attack_result([x_t], premise, predictor, orig_label, batch_size)) > 0 and (
                        num_changed == 1):
                    change_idx = 0
                    for i in range(len(text_ls)):
                        if text_ls[i] != x_t[i]:
                            change_idx = i
                            break
                    idx = word2idx[text_ls[change_idx]]
                    res = list(zip(*(cos_sim[idx])))
                    best_attack = x_t[:]
                    best_sim = x_t_sim
                    for widx in res[1]:
                        w = idx2word[widx]
                        x_t[change_idx] = w
                        pr = get_attack_result([x_t], premise, predictor, orig_label, batch_size)
                        sim = calc_sim(text_ls, [x_t], -1, sim_score_window, sim_predictor)[0]
                        qrs += 1
                        if np.sum(pr) > 0:
                            if sim >= best_sim:
                                best_attack = x_t[:]
                                best_sim = calc_sim(text_ls, [x_t], -1, sim_score_window, sim_predictor)[0]
                    return ' '.join(best_attack), 1, len(changed_indices), \
                        orig_label, torch.argmax(
                        predictor({'premises': [premise], 'hypotheses': [best_attack]})), qrs, best_sim, random_sim

                if np.sum(get_attack_result([x_t], premise, predictor, orig_label, batch_size)) > 0 and (
                        x_t_sim >= best_sim or False):
                    qrs += 1
                    best_attack = x_t[:]
                    best_sim = x_t_sim

                nonzero_ele = []
                
                perturb_word_idx_list = []
                ni = 0
                
                for idx in words_perturb_doc_idx:
                    if text_ls[idx]!=x_t[idx]:
                        nonzero_ele.append(ni)
                    ni+=1
                l2s = []
                for j in range(len(nonzero_ele)):
                    x_t_orig_word = x_t[synonyms_all[nonzero_ele[j]][0]]
                    orig_word = text_ls[synonyms_all[nonzero_ele[j]][0]]
                    v1 = np.array([float(num) for num in embed_content[word_idx_dict[x_t_orig_word]].strip().split()[1:]])
                    v2 = np.array([float(num) for num in embed_content[word_idx_dict[orig_word]].strip().split()[1:]])
                    sim_ = cos_sim_compute(v1,v2)
                    distance = 1+sim_
                    l2s.append(distance)
                
                p = torch.softmax(torch.tensor(l2s),dim=0).numpy()
                p /= p.sum()


                perturb_word_idx_list = np.random.choice(nonzero_ele,len(nonzero_ele),replace=False,p=p)

                x_tilde = text_ls[:]

                for perturb_word_idx in perturb_word_idx_list:
                    x_t_orig_word = x_t[synonyms_all[perturb_word_idx][0]]
                    orig_word = text_ls[synonyms_all[perturb_word_idx][0]]
                    ad_replacement = []
                    n_samples = []
                    while len(n_samples) < n_sample:
                        syn_idx = random.randint(0, 49)
                        if syn_idx not in n_samples:
                            n_samples.append(syn_idx)

                    for _ in range(n_sample):
                        x_t_tmp = x_t[:]
                        syn_idx = n_samples[_]
                        replacement = synonyms_all[perturb_word_idx][1][syn_idx]
                        x_t_tmp[synonyms_all[perturb_word_idx][0]] = replacement
                        if np.sum(get_attack_result([x_t_tmp],premise, predictor, orig_label, batch_size)) > 0:
                            sim_tmp = calc_sim(text_ls, [x_t_tmp], -1, sim_score_window, sim_predictor)[0]
                            if sim_tmp > best_sim:
                                best_attack = x_t_tmp[:]
                                best_sim = sim_tmp
                                best_qrs = qrs
                                for ksp in qrs_stopp:
                                        if qrs<=ksp:
                                            optim_step[ksp]=[best_sim," ".join(best_attack),best_qrs,np.sum(get_attack_result([best_attack],premise, predictor, orig_label, batch_size))>0]
                                wbcount = 0
                            ad_replacement.append((replacement, sim_tmp,syn_idx))

                        qrs += 1
                        if qrs > qrs_limits:
                            return ' '.join(best_attack), 1, len(changed_indices), \
                                orig_label, 1, qrs, best_sim, random_sim
                    
                    if len(ad_replacement) != 0:
                        ad_replacement = sorted(ad_replacement, key=lambda x: x[1],reverse=True)
                        condi_replacement = ad_replacement[0][0]
                        condi_replacement_idx = word2idx[condi_replacement]
                        base_sim = ad_replacement[0][1]
                        x_t_base = x_t[:]
                        x_t_base[synonyms_all[perturb_word_idx][0]] = condi_replacement

                        res = list(zip(*(cos_sim[condi_replacement_idx])))
                        condi_replacements = []
                        for i, j in zip(res[0], res[1]):
                            if i>k_threshold and j!=condi_replacement_idx:
                                condi_replacements.append((j,i,np.array([float(num) for num in embed_content[word_idx_dict[idx2word[j]]].strip().split()[1:]])))


                        best_rep = ad_replacement[0][0]

                        cds = random.sample(condi_replacements,k_sample)
                        vec_ls = []
                        condi_vec = np.array([float(num) for num in embed_content[condi_replacement_idx].strip().split()[1:]])
                        sim_ls = []
                        for ite in cds:
                            cd, similarity,vec = ite
                            cdw = idx2word[cd]
                            x_t_tmp = x_t[:]
                            x_t_tmp[synonyms_all[perturb_word_idx][0]] = cdw
                            sim_tmp = calc_sim(text_ls, [x_t_tmp], -1, sim_score_window, sim_predictor)[0]
                            inc = sim_tmp - base_sim
                            vec_ls.append(np.array(vec) - condi_vec)
                            sim_ls.append(inc)
                        sim_ls = np.array(sim_ls)
                        vec_ls = np.array(vec_ls)
                        fn(sim_ls)
                        estimate_vec = 0
                        for e_idx in range(len(sim_ls)):
                            estimate_vec += sim_ls[e_idx] * vec_ls[e_idx]

                        candi_2 = []
                        
                        contextual_replacements = []
                        res = list(zip(*(cos_sim[word2idx[orig_word]])))
                        for i,j in zip(res[0],res[1]):
                            contextual_replacements.append((j,i,np.array([float(num) for num in embed_content[word_idx_dict[idx2word[j]]].strip().split()[1:]])))
                        for ite in contextual_replacements:
                            cd, similarity, vec = ite
                            candi_2.append((cd,cos_sim_compute(estimate_vec,np.array(vec)-condi_vec)))
                        candi_2 = sorted(candi_2,key=lambda x:x[-1],reverse=True)

                        for ite in candi_2:
                            cd,similarity = ite
                            x_t_tmp = x_t[:]
                            x_t_tmp[synonyms_all[perturb_word_idx][0]] = idx2word[cd]
                            pr = get_attack_result([x_t_tmp],premise, predictor, orig_label, batch_size)
                            qrs+=1

                            if np.sum(pr)>0:
                                sim_tmp = calc_sim(text_ls, [x_t_tmp], -1, sim_score_window, sim_predictor)[0]
                                if sim_tmp > best_sim:
                                    best_attack = x_t_tmp[:]
                                    best_sim = sim_tmp
                                    best_qrs = qrs
                                    wbcount=0
                                    for ksp in qrs_stopp:
                                        if qrs<=ksp:
                                            optim_step[ksp]=[best_sim," ".join(best_attack),best_qrs,np.sum(get_attack_result([best_attack],premise, predictor, orig_label, batch_size))>0]
                                best_rep = idx2word[cd]
                                break
                            if qrs>qrs_limits:
                                if np.sum(pr)>0:
                                    sim_tmp = calc_sim(text_ls, [x_t_tmp], -1, sim_score_window, sim_predictor)[0]
                                    if sim_tmp > best_sim:
                                        best_attack = x_t_tmp[:]
                                        best_sim = sim_tmp
                                        best_qrs = qrs
                                        wbcount=0
                                        for ksp in qrs_stopp:
                                            if qrs<=ksp:
                                                optim_step[ksp]=[best_sim," ".join(best_attack),best_qrs,np.sum(get_attack_result([best_attack],premise, predictor, orig_label, batch_size))>0]

                                return ' '.join(best_attack), 1, len(changed_indices), \
                                        orig_label, 1, qrs, best_sim, random_sim
                        x_tilde[synonyms_all[perturb_word_idx][0]] = best_rep
                    else:
                        x_tilde[synonyms_all[perturb_word_idx][0]] = x_t_orig_word

                    pr = get_attack_result([x_tilde],premise, predictor, orig_label, batch_size)
                    qrs += 1
                    if np.sum(pr) > 0:
                        sim_new = calc_sim(text_ls, [x_tilde], -1, sim_score_window, sim_predictor)[0]
                        if (sim_new > best_sim) and (
                                np.sum(get_attack_result([x_tilde],premise, predictor, orig_label, batch_size)) > 0):
                            best_attack = x_tilde[:]
                            best_sim = sim_new
                            best_qrs = qrs
                            wbcount = 0
                            for ksp in qrs_stopp:
                                if qrs<=ksp:
                                    optim_step[ksp]=[best_sim," ".join(best_attack),best_qrs,np.sum(get_attack_result([best_attack],premise, predictor, orig_label, batch_size))>0]
                    if qrs > qrs_limits:
                        return ' '.join(best_attack), 1, len(changed_indices), \
                            orig_label, 1, qrs, best_sim, random_sim
                    if np.sum(pr) > 0:
                        break

                if np.sum(pr) > 0:
                    sim_new = calc_sim(text_ls, [x_tilde], -1, sim_score_window, sim_predictor)[0]
                    if (sim_new > best_sim) and (
                            np.sum(get_attack_result([x_tilde],premise, predictor, orig_label, batch_size)) > 0):
                        best_attack = x_tilde[:]
                        best_sim = sim_new
                        best_qrs = qrs
                        wbcount = 0
                        for ksp in qrs_stopp:
                            if qrs<=ksp:
                                optim_step[ksp]=[best_sim," ".join(best_attack),best_qrs,np.sum(get_attack_result([best_attack],premise, predictor, orig_label, batch_size))>0]
                else:

                    while True:
                        changed_tt = []
                        for i in range(len(x_t)):
                            if x_tilde[i]!=x_t[i]:
                                x_tilde_base = x_tilde[:]
                                x_tilde_base[i] = x_t[i]

                                sim_tmp = calc_sim(text_ls, [x_tilde_base], -1, sim_score_window, sim_predictor)[0]
                                changed_tt.append((i,sim_tmp))
                        changed_tt = sorted(changed_tt,key=lambda x:x[-1],reverse=True)
                        x_tilde[changed_tt[0][0]] = x_t[changed_tt[0][0]]
                        if np.sum(get_attack_result([x_tilde],premise, predictor, orig_label, batch_size)) > 0:
                            qrs+=1
                            sim_new = calc_sim(text_ls, [x_tilde], -1, sim_score_window, sim_predictor)[0]
                            if sim_new > best_sim:
                                best_sim = sim_new
                                best_attack = x_tilde[:]
                                wbcount = 0
                                for ksp in qrs_stopp:
                                    if qrs<=ksp:
                                        optim_step[ksp]=[best_sim," ".join(best_attack),best_qrs,np.sum(get_attack_result([best_attack],premise, predictor, orig_label, batch_size))>0]
                            if qrs > qrs_limits:
                                return ' '.join(best_attack), 1, len(changed_indices), \
                                    orig_label, 1, qrs, best_sim, random_sim
                            break
                        qrs+=1

            sim = best_sim
            max_changes = 0
            for i in range(len(text_ls)):
                if text_ls[i] != best_attack[i]:
                    max_changes += 1

            return ' '.join(best_attack), max_changes, len(changed_indices), \
                orig_label, torch.argmax(
                predictor({'premises': [premise], 'hypotheses': [best_attack]})), qrs, sim, random_sim

        else:
            print("Not Found")
            return '', 0, 0, orig_label, orig_label, 0, 0, 0







def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--dataset_path",
                        type=str,
                        required=True,
                        help="Which dataset to attack.")
    parser.add_argument("--target_model",
                        type=str,
                        required=True,
                        choices=['infersent', 'esim', 'bert'],
                        help="Target models for text classification: fasttext, charcnn, word level lstm "
                             "For NLI: InferSent, ESIM, bert-base-uncased")
    parser.add_argument("--output_dir",
                        type=str,
                        default='adv_results',
                        help="The output directory where the attack results will be written.")
    parser.add_argument("--target_model_path",
                        type=str,
                        required=True,
                        help="pre-trained target model path")
    parser.add_argument("--word_embeddings_path",
                        type=str,
                        default='',
                        help="path to the word embeddings for the target model")
    parser.add_argument("--counter_fitting_embeddings_path",
                        type=str,
                        default="counter-fitted-vectors.txt",
                        help="path to the counter-fitting embeddings we used to find synonyms")
    parser.add_argument("--counter_fitting_cos_sim_path",
                        type=str,
                        default='',
                        help="pre-compute the cosine similarity scores based on the counter-fitting embeddings")
    parser.add_argument("--USE_cache_path",
                        type=str,
                        required=True,
                        help="Path to the USE encoder cache.")

    ## Model hyperparameters
    parser.add_argument("--sim_score_window",
                        default=310,
                        type=int,
                        help="Text length or token number to compute the semantic similarity score")
    parser.add_argument("--import_score_threshold",
                        default=-1.,
                        type=float,
                        help="Required mininum importance score.")
    parser.add_argument("--sim_score_threshold",
                        default=0.47,
                        type=float,
                        help="Required minimum semantic similarity score.")
    parser.add_argument("--synonym_num",
                        default=50,
                        type=int,
                        help="Number of synonyms to extract")
    parser.add_argument("--batch_size",
                        default=32,
                        type=int,
                        help="Batch size to get prediction")
    parser.add_argument("--data_size",
                        default=1000,
                        type=int,
                        help="Data size to create adversaries")
    parser.add_argument("--perturb_ratio",
                        default=0.,
                        type=float,
                        help="Whether use random perturbation for ablation study")
    parser.add_argument("--target_dataset",
                        default="imdb",
                        type=str,
                        help="Dataset Name")
    parser.add_argument("--fuzz",
                        default=0,
                        type=int,
                        help="Word Pruning Value")
    parser.add_argument("--top_k_words",
                        default=1000000,
                        type=int,
                        help="Top K Words")
    parser.add_argument("--allowed_qrs",
                        default=1000000,
                        type=int,
                        help="Allowerd qrs")

    args = parser.parse_args()
    log_file = "results_nli_hard_label/" + args.target_model + "/" + args.target_dataset + "/log.txt"
    result_file = "results_nli_hard_label/" + args.target_model + "/" + args.target_dataset + "/results_final.csv"
    Path(result_file).mkdir(parents=True, exist_ok=True)
    Path(log_file).mkdir(parents=True, exist_ok=True)
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        print("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    else:
        os.makedirs(args.output_dir, exist_ok=True)

    # get data to attack, fetch first [args.data_size] data samples for adversarial attacking
    data = read_data(args.dataset_path, data_size=args.data_size, target_model=args.target_model)
    print("Data import finished!")

    # construct the model
    print("Building Model...")
    if args.target_model == 'esim':
        model = NLI_infer_ESIM(args.target_model_path,
                               args.word_embeddings_path,
                               batch_size=args.batch_size)
    elif args.target_model == 'infersent':
        model = NLI_infer_InferSent(args.target_model_path,
                                    args.word_embeddings_path,
                                    data=data,
                                    batch_size=args.batch_size)
    else:
        model = NLI_infer_BERT(args.target_model_path)
    predictor = model.text_pred
    print("Model built!")

    # prepare synonym extractor
    # build dictionary via the embedding file
    print("Building vocab...")
    idx2word = {}
    word2idx = {}
    sim_lis = []
    with open(args.counter_fitting_embeddings_path, 'r') as ifile:
        for line in ifile:
            word = line.split()[0]
            if word not in idx2word:
                idx2word[len(idx2word)] = word
                word2idx[word] = len(idx2word) - 1

    # for cosine similarity matrix
    print("Building cos sim matrix...")
    if args.counter_fitting_cos_sim_path:
        # load pre-computed cosine similarity matrix if provided
        print('Load pre-computed cosine similarity matrix from {}'.format(args.counter_fitting_cos_sim_path))
        with open(args.counter_fitting_cos_sim_path, "rb") as fp:
            sim_lis = pickle.load(fp)
    else:
        # calculate the cosine similarity matrix
        print('Start computing the cosine similarity matrix!')
        embeddings = []
        with open(args.counter_fitting_embeddings_path, 'r') as ifile:
            for line in ifile:
                embedding = [float(num) for num in line.strip().split()[1:]]
                embeddings.append(embedding)
        embeddings = np.array(embeddings)
        print(embeddings.T.shape)
        norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = np.asarray(embeddings / norm, "float64")
        cos_sim = np.dot(embeddings, embeddings.T)
    print("Cos sim import finished!")

    # build the semantic similarity module
    use = USE(args.USE_cache_path)

    # start attacking

    changed_rates = []

    stop_words_set = criteria.get_stopwords()

    db_sims = []
    db_cr = []

    for idx, premise in enumerate(data['premises']):
        if idx % 100 == 0:
            print(np.mean(changed_rates))

            print('{} samples out of {} have been finished!'.format(idx, args.data_size))

        hypothese, true_label = data['hypotheses'][idx], data['labels'][idx]

        random_text, random_qrs, orig_label, flag = random_attack(args.fuzz, args.top_k_words, args.allowed_qrs,
                                                                  idx, hypothese, premise, true_label, predictor,
                                                                  stop_words_set,
                                                                  word2idx, idx2word, sim_lis, sim_predictor=use,
                                                                  sim_score_threshold=args.sim_score_threshold,
                                                                  import_score_threshold=args.import_score_threshold,
                                                                  sim_score_window=args.sim_score_window,
                                                                  synonym_num=args.synonym_num,
                                                                  batch_size=args.batch_size)

        if flag:
            print("Attacked ", idx)
            optim_step={}
            for kp in [10,20,30,40,50,60,70,80,90,100,200,300,400,500,600,700,800,900,1000]:
                optim_step[kp] = []

            new_text, num_changed, random_changed, orig_label, \
            new_label, num_queries, sim, random_sim = attack(args.fuzz, args.top_k_words,optim_step, args.allowed_qrs,
                                                             idx, hypothese,random_text, premise, true_label,orig_label, predictor,
                                                             stop_words_set,
                                                             word2idx, idx2word, sim_lis, sim_predictor=use,
                                                             sim_score_threshold=args.sim_score_threshold,
                                                             import_score_threshold=args.import_score_threshold,
                                                             sim_score_window=args.sim_score_window,
                                                             n_sample = 5,
                                                             synonym_num=args.synonym_num,
                                                             batch_size=args.batch_size,
                                                             embed_func=args.counter_fitting_embeddings_path)

            num_changed = 0
            for i,j in zip(new_text.split(),hypothese):
                if i!=j:
                    num_changed += 1
            db_sims.append(float(sim))
            db_cr.append(num_changed/len(hypothese))



            print(f"my sims {np.mean(db_sims)}")
            print(f"my changes rate {np.mean(db_cr)}")


if __name__ == "__main__":
    main()
