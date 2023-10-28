import argparse
import os
import numpy as np
from pathlib import Path
import json
from scipy import spatial
from scipy.special import softmax

np.random.seed(1234)
import pickle
import dataloader
from train_classifier import Model
from itertools import zip_longest
import criteria
import random

random.seed(0)
import csv
import math
import sys
import pdb


# csv.field_size_limit(sys.maxsize)
csv.field_size_limit(2147483647)

# import tensorflow as tf
# To make tf 2.0 compatible with tf1.0 code, we disable the tf2.0 functionalities
# tf.compat.v1.disable_eager_execution()
import tensorflow_hub as hub
import tensorflow as tf
# tf.disable_v2_behavior()
import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
from collections import defaultdict
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SequentialSampler, TensorDataset

from BERT.tokenization import BertTokenizer
from BERT.modeling import BertForSequenceClassification, BertConfig

tf.compat.v1.disable_eager_execution()
gpus = tf.config.experimental.list_physical_devices('GPU')
np.random.seed(0)
random.seed(0)

if gpus:
    # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024 * 1.5)])
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


    
    # def l2_distance(self,sents1,sents2):
        # sts_encode1 = self.embed(tf.Variable([sents1],shape=[1,],dtype=tf.string))
        # sts_encode2 = self.embed(tf.Variable([sents2],shape=[1,],dtype=tf.string))
        
        # distance = tf.norm(sts_encode1-sts_encode2,axis=1)
        # pdb.set_trace()
        # tf.initialize_all_variables().run() 
        # with tf.Session().as_defualt():
        #     sts_encode1 = sts_encode1.eval()
            
        # pdb.set_trace()
        
        # return distance


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


class NLI_infer_BERT(nn.Module):
    def __init__(self,
                 pretrained_dir,
                 nclasses,
                 max_seq_length=128,
                 batch_size=32):
        super(NLI_infer_BERT, self).__init__()
        self.model = BertForSequenceClassification.from_pretrained(pretrained_dir, num_labels=nclasses).cuda()

        # construct dataset loader
        self.dataset = NLIDataset_BERT(pretrained_dir, max_seq_length=max_seq_length, batch_size=batch_size)

    def text_pred(self, text_data, batch_size=32):
        # Switch the model to eval mode.
        self.model.eval()

        # transform text data into indices and create batches
        dataloader = self.dataset.transform_text(text_data, batch_size=batch_size)

        probs_all = []
        # print (text_data)
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

    def convert_examples_to_features(self, examples, max_seq_length, tokenizer):
        """Loads a data file into a list of `InputBatch`s."""

        features = []
        for (ex_index, text_a) in enumerate(examples):
            tokens_a = tokenizer.tokenize(' '.join(text_a))

            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

            tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
            segment_ids = [0] * len(tokens)

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

    def transform_text(self, data, batch_size=32):
        # transform data into seq of embeddings
        eval_features = self.convert_examples_to_features(data,
                                                          self.max_seq_length, self.tokenizer)

        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)

        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=batch_size)

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
    # Calculate semantic similarity using USE.
    semantic_sims = \
        sim_predictor.semantic_sim([' '.join(text_ls[text_range_min:text_range_max])],
                                   list(map(lambda x: ' '.join(x[text_range_min:text_range_max]), new_texts)))[0]

    return semantic_sims


def get_attack_result(new_text, predictor, orig_label, batch_size,hash_qrs):
    # new_text = "turkey unlikely to join eu before 2015 commissioner verheugen \\( afp \\) afp turkey is unlikely to join the euro union before 2015 , eu growth commissioner guenter verheugen said in an interview"
    # assert len(new_text)==1
    if " ".join(new_text[0]) in hash_qrs:
        return hash_qrs[" ".join(new_text[0])],1
    
    new_probs = predictor(new_text, batch_size=batch_size)
    pr = (orig_label != torch.argmax(new_probs, dim=-1)).data.cpu().numpy()
    hash_qrs[" ".join(new_text[0])] = pr
    return pr,1


def random_attack(top_k_words, text_ls, true_label,
                  predictor, word2idx, idx2word, cos_sim, sim_score_window=15,
                  batch_size=32):
    # first check the prediction of the original text
    orig_probs = predictor([text_ls]).squeeze()
    orig_label = torch.argmax(orig_probs)
    orig_prob = orig_probs.max()
    if true_label != orig_label:
        return text_ls, 1, orig_label, False
    else:

        # pos_ls = criteria.get_pos(text_ls)
        len_text = len(text_ls)
        if len_text < sim_score_window:
            sim_score_threshold = 0.1  # shut down the similarity thresholding function
        half_sim_score_window = (sim_score_window - 1) // 2
        num_queries = 1
        rank = {}

        words_perturb = []
        pos_ls = criteria.get_pos(text_ls)
        pos_pref = ["ADJ", "ADV", "VERB", "NOUN"]
        for pos in pos_pref:
            for i in range(len(pos_ls)):
                if pos_ls[i] == pos and len(text_ls[i]) > 2:
                    words_perturb.append((i, text_ls[i]))

        random.shuffle(words_perturb)
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

            pr = get_attack_result([random_text], predictor, orig_label, batch_size)

            qrs += 1
            th += 1
            if th > len_text:
                break
            if np.sum(pr) > 0:
                flag = 1
                break
        old_qrs = qrs
        # If adversarial text is not yet generated try to substitute more words than 30%.
        while qrs < old_qrs + 2500 and flag == 0:
            random_text = text_ls[:]
            for j in range(len(synonyms_all)):
                idx = synonyms_all[j][0]
                syn = synonyms_all[j][1]
                random_text[idx] = random.choice(syn)
                if j >= len_text:
                    break
            pr = get_attack_result([random_text], predictor, orig_label, batch_size)
            qrs += 1
            if np.sum(pr) > 0:
                flag = 1
                break

        if flag == 1:
            return random_text, qrs, orig_label, True
        else:
            return text_ls, qrs, orig_label, False


def get_pert_rate(text, ade):
    text_ls = text.split()
    ade_ls = ade.split()
    changed = 0
    for i, j in zip(text_ls, ade_ls):
        if i != j:
            changed += 1
    return changed / len(text_ls)


l2s = lambda l: " ".join(l)


def fn(sim_ls):
    abs_sum = np.sum([np.abs(i) for i in sim_ls])
    for j in range(len(sim_ls)):
        sim_ls[j] = sim_ls[j]/abs_sum

def softmax(x):
    x = np.array(x)
    y = np.exp(x - np.max(x))
    f_x = y / np.sum(np.exp(x))
    return f_x


def cos_sim_compute(x,y):
    x = torch.FloatTensor(x)
    y = torch.FloatTensor(y)
    return float(torch.cosine_similarity(x,y,dim=0).numpy())

def attack(fuzz_val,optim_step, orig_label, top_k_words, qrs, sample_index, text_ls, random_text_, true_label,
             predictor, stop_words_set, word2idx, idx2word, cos_sim, sim_predictor=None,
             import_score_threshold=-1., sim_score_threshold=0.5, sim_score_window=15, synonym_num=50,
             batch_size=32, embed_func='', n_sample=5,k_sample=5,k_threshold=0.8, qrs_limits=1000,qrs_stopp=[100,200,300,400,500,600,700,800,900,1000]):

    random_text = random_text_[:]
    word_idx_dict = {}
    with open(embed_func, 'r') as ifile:
        for index, line in enumerate(ifile):
            word = line.strip().split()[0]
            word_idx_dict[word] = index

    embed_file = open(embed_func)
    embed_content = embed_file.readlines()

    words_perturb = [] 
    pos_ls = criteria.get_pos(text_ls)
    pos_pref = ["ADJ", "ADV", "VERB", "NOUN"]
    for pos in pos_pref:
        for i in range(len(pos_ls)):
            if pos_ls[i] == pos and len(text_ls[i]) > 2:
                words_perturb.append((i, text_ls[i]))

    random.shuffle(words_perturb)
    words_perturb = words_perturb[:top_k_words]

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
                synonyms_all.append((idx,synonyms))
                synonyms_dict[word] = synonyms

    qrs = old_qrs
    flag = 1
    best_qrs = 0
    if flag == 1:
        
        changed = 0
        for i in range(len(text_ls)):
            if text_ls[i] != random_text[i]:
                changed += 1

        changed_indices = []
        num_changed = 0
        for i in range(len(text_ls)):
            if text_ls[i] != random_text[i]:
                changed_indices.append(i)
                num_changed += 1

        random_sim = calc_sim(text_ls, [random_text], -1, sim_score_window, sim_predictor)[0]
        
        best_sim = random_sim
        best_attack = random_text[:]
        best_qrs = qrs
        for kp in qrs_stopp:
            if qrs<=kp:
                optim_step[kp] = [best_sim," ".join(best_attack),qrs,np.sum(get_attack_result([best_attack], predictor, orig_label, batch_size,hash_qrs)[0])>0]

        random_sim = calc_sim(text_ls, [random_text], -1, sim_score_window, sim_predictor)[0]
        best_sim = random_sim
        best_attack = random_text[:]
        x_t = random_text[:]
        if num_changed == 1:

            change_idx = 0
            for i in range(len(text_ls)):
                if text_ls[i] != x_t[i]:
                    change_idx = i
                    break
            idx = word2idx[text_ls[change_idx]]
            res = list(zip(*(cos_sim[idx])))
            for widx in res[1]:
                w = idx2word[widx]
                x_t[change_idx] = w
                pr,vqr = get_attack_result([x_t], predictor, orig_label, batch_size,hash_qrs)
                sim = calc_sim(text_ls, [x_t], -1, sim_score_window, sim_predictor)[0]
                qrs += vqr
                if np.sum(pr) > 0 and round(sim,3) >= round(best_sim,3):
                    best_attack = x_t[:]
                    best_sim = calc_sim(text_ls, [x_t], -1, sim_score_window, sim_predictor)[0]
                    best_qrs = qrs
                    for kp in qrs_stopp:
                        if qrs<=kp:
                            optim_step[kp] = [best_sim," ".join(best_attack),best_qrs,np.sum(get_attack_result([best_attack], predictor, orig_label, batch_size,hash_qrs)[0])>0]

            return ' '.join(best_attack), 1, 1, \
                   orig_label, 1, qrs, best_sim, random_sim, best_qrs

        num_changed = 0
        for i in range(len(text_ls)):
            if text_ls[i] != x_t[i]:
                num_changed += 1

        best_attack = random_text[:]
        best_sim = calc_sim(text_ls, [best_attack], -1, sim_score_window, sim_predictor)[0]

        stack = [random_text[:]]
        stack_str = []
        stack_over = []
        way_back_num = 3

        x_tilde = random_text[:]
        wbcount = 0
        for t in range(100):

            x_t = x_tilde[:]
            x_t_str = " ".join(x_t)
            if " ".join(x_t) not in stack_over and " ".join(x_t) not in stack_str:
                pr,vqr = get_attack_result([x_t], predictor, orig_label, batch_size,hash_qrs)
                qrs += vqr
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
                    way_back_num += 5
                wbcount = 0
            
            
            while True and t==0:
                choices = []
                for i in range(len(text_ls)):
                    if x_t[i] != text_ls[i]:
                        new_text = x_t[:]
                        new_text[i] = text_ls[i]
                        semantic_sims = calc_sim(text_ls, [new_text], -1, sim_score_window, sim_predictor)
                        
                        pr,vqr = get_attack_result([new_text], predictor, orig_label, batch_size,hash_qrs)
                        qrs += vqr
                        if np.sum(pr) > 0:
                            choices.append((i, semantic_sims[0]))


                if len(choices) > 0:
                    choices.sort(key=lambda x: x[1])
                    choices.reverse()
                    for i in range(len(choices)):
                        new_text = x_t[:]
                        new_text[choices[i][0]] = text_ls[choices[i][0]]
                        pr,vqr = get_attack_result([new_text], predictor, orig_label, batch_size,hash_qrs)
                        qrs += vqr
                        if pr[0] == 0:
                            break
                        x_t[choices[i][0]] = text_ls[choices[i][0]]

                if len(choices) == 0:
                    break

            while True and t!=0:
                choices = []

                for i in range(len(text_ls)):
                    if x_t[i] != text_ls[i]:
                        new_text = x_t[:]
                        new_text[i] = text_ls[i]
                        semantic_sims = calc_sim(text_ls, [new_text], -1, sim_score_window, sim_predictor)
                        choices.append((i,semantic_sims[0]))
                
                choices.sort(key=lambda x: x[1])
                choices.reverse()
                flag = True
                for i in range(len(choices)):
                    new_text = x_t[:]
                    new_text[choices[i][0]] = text_ls[choices[i][0]]
                    pr,vqr = get_attack_result([new_text], predictor, orig_label, batch_size,hash_qrs)
                    qrs += vqr
                    if np.sum(pr)>0:
                        x_t[choices[i][0]] = text_ls[choices[i][0]]
                        sims = calc_sim(text_ls, [x_t], -1, sim_score_window, sim_predictor)[0]
                        if round(sims,3) >= round(best_sim,3):
                            best_attack=x_t[:]
                            best_sim=sims
                            best_qrs=qrs
                            wbcount = 0
                            for ksp in qrs_stopp:
                                if qrs<=ksp:
                                    optim_step[ksp]=[best_sim," ".join(best_attack),best_qrs,np.sum(get_attack_result([best_attack], predictor, orig_label, batch_size,hash_qrs)[0])>0]
                        flag = False
                        break
                    if qrs > qrs_limits:
                        return ' '.join(best_attack), 1, len(changed_indices), \
                            orig_label, 1, qrs, best_sim, random_sim, best_qrs
                if flag:
                    break

            num_changed = 0
            for i in range(len(text_ls)):
                if text_ls[i] != x_t[i]:
                    num_changed += 1

            x_t_sim = calc_sim(text_ls, [x_t], -1, sim_score_window, sim_predictor)[0]

            if np.sum(
                    get_attack_result([x_t], predictor, orig_label, batch_size,hash_qrs)[0]) > 0 and (
                    num_changed == 1):
                change_idx = 0
                for i in range(len(text_ls)):
                    if text_ls[i] != x_t[i]:
                        change_idx = i
                        break
                idx = word2idx[text_ls[change_idx]]
                res = list(zip(*(cos_sim[idx])))
                for widx in res[1]:
                    w = idx2word[widx]
                    x_t[change_idx] = w
                    pr,vqr = get_attack_result([x_t], predictor, orig_label, batch_size,hash_qrs)
                    sim = calc_sim(text_ls, [x_t], -1, sim_score_window, sim_predictor)[0]
                    qrs += vqr
                    
                    if np.sum(pr) > 0 and round(sim,3) >= round(best_sim,3):
                        best_sim = calc_sim(text_ls, [x_t], -1, sim_score_window, sim_predictor)[0]
                        best_attack = x_t[:]
                        best_qrs = qrs
                        wbcount = 0
                        for ksp in qrs_stopp:
                            if qrs<=ksp:
                                optim_step[ksp]=[best_sim," ".join(best_attack),best_qrs,np.sum(get_attack_result([best_attack], predictor, orig_label, batch_size,hash_qrs)[0])>0]
                    if qrs > qrs_limits:
                        return ' '.join(best_attack), 1, len(changed_indices), \
                               orig_label, 1, qrs, best_sim, random_sim,  best_qrs
                return ' '.join(best_attack), 1, len(changed_indices), \
                       orig_label, 1, qrs, best_sim, random_sim, best_qrs

            if np.sum(
                    get_attack_result([x_t], predictor, orig_label, batch_size,hash_qrs)[0]) > 0 and (
                    round(x_t_sim,3) > round(best_sim,3)):
                best_attack = x_t[:]
                best_sim = x_t_sim
                best_qrs = qrs
                wbcount = 0
                for ksp in qrs_stopp:
                    if qrs<=ksp:
                        optim_step[ksp]=[best_sim," ".join(best_attack),best_qrs,np.sum(get_attack_result([best_attack], predictor, orig_label, batch_size,hash_qrs)[0])>0]
                wbcount = 0


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

                # pdb.set_trace()
                while len(n_samples) < n_sample:
                    syn_idx = random.randint(0, 49)
                    if syn_idx not in n_samples:
                        n_samples.append(syn_idx)

                for _ in range(n_sample):
                    x_t_tmp = x_t[:]
                    syn_idx = n_samples[_]
                    replacement = synonyms_all[perturb_word_idx][1][syn_idx]
                    x_t_tmp[synonyms_all[perturb_word_idx][0]] = replacement
                    pr_v,vqr = get_attack_result([x_t_tmp], predictor, orig_label, batch_size,hash_qrs)
                    if np.sum(pr_v) > 0:
                        sim_tmp = calc_sim(text_ls, [x_t_tmp], -1, sim_score_window, sim_predictor)[0]
                        if round(sim_tmp,3) >= round(best_sim,3):
                            best_attack = x_t_tmp[:]
                            best_sim = sim_tmp
                            best_qrs = qrs
                            wbcount = 0
                            for ksp in qrs_stopp:
                                    if qrs<=ksp:
                                        optim_step[ksp]=[best_sim," ".join(best_attack),best_qrs,np.sum(get_attack_result([best_attack], predictor, orig_label, batch_size,hash_qrs)[0])>0]
                            wbcount = 0
                        ad_replacement.append((replacement, sim_tmp,syn_idx))

                    qrs += vqr
                    if qrs > qrs_limits:
                        return ' '.join(best_attack), 1, len(changed_indices), \
                               orig_label, 1, qrs, best_sim, random_sim, best_qrs\
                
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
                        pr,vqr = get_attack_result([x_t_tmp], predictor, orig_label, batch_size,hash_qrs)
                        qrs+=vqr

                        if np.sum(pr)>0:
                            sim_tmp = calc_sim(text_ls, [x_t_tmp], -1, sim_score_window, sim_predictor)[0]
                            if round(sim_tmp,3) >= round(best_sim,3):
                                best_attack = x_t_tmp[:]
                                best_sim = sim_tmp
                                best_qrs = qrs
                                wbcount=0
                                for ksp in qrs_stopp:
                                    if qrs<=ksp:
                                        optim_step[ksp]=[best_sim," ".join(best_attack),best_qrs,np.sum(get_attack_result([best_attack], predictor, orig_label, batch_size,hash_qrs)[0])>0]
                            best_rep = idx2word[cd]
                            break
                        if qrs>qrs_limits:
                            if np.sum(pr)>0:
                                sim_tmp = calc_sim(text_ls, [x_t_tmp], -1, sim_score_window, sim_predictor)[0]
                                if round(sim_tmp,3) >= round(best_sim,3):
                                    best_attack = x_t_tmp[:]
                                    best_sim = sim_tmp
                                    best_qrs = qrs
                                    wbcount=0
                                    for ksp in qrs_stopp:
                                        if qrs<=ksp:
                                            optim_step[ksp]=[best_sim," ".join(best_attack),best_qrs,np.sum(get_attack_result([best_attack], predictor, orig_label, batch_size,hash_qrs)[0])>0]

                            return ' '.join(best_attack), 1, len(changed_indices), \
                                    orig_label, 1, qrs, best_sim, random_sim, best_qrs
                    x_tilde[synonyms_all[perturb_word_idx][0]] = best_rep
                else:
                    x_tilde[synonyms_all[perturb_word_idx][0]] = x_t_orig_word

                pr,vqr = get_attack_result([x_tilde], predictor, orig_label, batch_size,hash_qrs)
                qrs += vqr
                if np.sum(pr) > 0:
                    sim_new = calc_sim(text_ls, [x_tilde], -1, sim_score_window, sim_predictor)[0]
                    if (round(sim_new,3) >= round(best_sim,3)) and (
                            np.sum(get_attack_result([x_tilde], predictor, orig_label, batch_size,hash_qrs)[0]) > 0):
                        best_attack = x_tilde[:]
                        best_sim = sim_new
                        best_qrs = qrs
                        wbcount = 0
                        for ksp in qrs_stopp:
                            if qrs<=ksp:
                                optim_step[ksp]=[best_sim," ".join(best_attack),best_qrs,np.sum(get_attack_result([best_attack], predictor, orig_label, batch_size,hash_qrs)[0])>0]
                if qrs > qrs_limits:
                    return ' '.join(best_attack), 1, len(changed_indices), \
                           orig_label, 1, qrs, best_sim, random_sim, best_qrs
                if np.sum(pr) > 0:
                    break

            if np.sum(pr) > 0:
                sim_new = calc_sim(text_ls, [x_tilde], -1, sim_score_window, sim_predictor)[0]
                if (round(sim_new,3) > round(best_sim,3)) and (
                        np.sum(get_attack_result([x_tilde], predictor, orig_label, batch_size,hash_qrs)[0]) > 0):
                    best_attack = x_tilde[:]
                    best_sim = sim_new
                    best_qrs = qrs
                    wbcount = 0
                    for ksp in qrs_stopp:
                        if qrs<=ksp:
                            optim_step[ksp]=[best_sim," ".join(best_attack),best_qrs,np.sum(get_attack_result([best_attack], predictor, orig_label, batch_size,hash_qrs)[0])>0]
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
                    if np.sum(get_attack_result([x_tilde], predictor, orig_label, batch_size,hash_qrs)[0]) > 0:
                        qrs+=1
                        sim_new = calc_sim(text_ls, [x_tilde], -1, sim_score_window, sim_predictor)[0]
                        if round(sim_new,3) >= round(best_sim,3):
                            best_sim = sim_new
                            best_attack = x_tilde[:]
                            wbcount = 0
                            for ksp in qrs_stopp:
                                if qrs<=ksp:
                                    optim_step[ksp]=[best_sim," ".join(best_attack),best_qrs,np.sum(get_attack_result([best_attack], predictor, orig_label, batch_size,hash_qrs)[0])>0]
                        if qrs > qrs_limits:
                            return ' '.join(best_attack), 1, len(changed_indices), \
                                orig_label, 1, qrs, best_sim, random_sim, best_qrs
                        break
                    qrs+=1

                
            if x_t_str == " ".join(x_tilde):
                wbcount += 1
            else:
                wbcount = 0

        sim = float(best_sim)
        max_changes = 0
        for i in range(len(text_ls)):
            if text_ls[i] != best_attack[i]:
                max_changes += 1

        return ' '.join(best_attack), max_changes, len(changed_indices), \
               orig_label, 1, qrs, sim, random_sim, best_qrs

    else:
        print("Not Found")
        return '', 0, 0, orig_label, orig_label, 0, 0, 0, 0

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--dataset_path",
                        type=str,
                        required=True,
                        help="Which dataset to attack.")
    parser.add_argument("--nclasses",
                        type=int,
                        default=2,
                        help="How many classes for classification.")
    parser.add_argument("--target_model",
                        type=str,
                        required=True,
                        choices=['wordLSTM', 'bert', 'wordCNN'],
                        help="Target models for text classification: fasttext, charcnn, word level lstm "
                             "For NLI: InferSent, ESIM, bert-base-uncased")
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
    parser.add_argument("--output_dir",
                        type=str,
                        default='adv_results',
                        help="The output directory where the attack results will be written.")

    ## Model hyperparameters
    parser.add_argument("--sim_score_window",
                        default=15,
                        type=int,
                        help="Text length or token number to compute the semantic similarity score")
    parser.add_argument("--import_score_threshold",
                        default=-1.,
                        type=float,
                        help="Required mininum importance score.")
    parser.add_argument("--sim_score_threshold",
                        default=0.7,
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
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="max sequence length for BERT target model")
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
    print("parser okk.")

    # get data to attack
    texts, labels = dataloader.read_corpus(args.dataset_path, csvf=False)
    data = list(zip(texts, labels))
    data = data[:args.data_size]  # choose how many samples for adversary
    print("Data import finished!")
    # pdb.set_trace()
    # construct the model
    print("Building Model...")
    if args.target_model == 'wordLSTM':
        model = Model(args.word_embeddings_path, nclasses=args.nclasses).cuda()
        checkpoint = torch.load(args.target_model_path, map_location='cuda:0')
        model.load_state_dict(checkpoint)
    elif args.target_model == 'wordCNN':
        model = Model(args.word_embeddings_path, nclasses=args.nclasses, hidden_size=150, cnn=True).cuda()
        checkpoint = torch.load(args.target_model_path, map_location='cuda:0')
        model.load_state_dict(checkpoint)
    elif args.target_model == 'bert':
        model = NLI_infer_BERT(args.target_model_path, nclasses=args.nclasses, max_seq_length=args.max_seq_length)
    predictor = model.text_pred
    print("Model built!")

    # prepare synonym extractor
    # build dictionary via the embedding file
    idx2word = {}
    word2idx = {}
    sim_lis = []

    print("Building vocab...")
    with open(args.counter_fitting_embeddings_path, 'r') as ifile:
        for line in ifile:
            word = line.split()[0]
            if word not in idx2word:
                idx2word[len(idx2word)] = word
                word2idx[word] = len(idx2word) - 1

    print("Building cos sim matrix...")
    if args.counter_fitting_cos_sim_path:
        # load pre-computed cosine similarity matrix if provided
        print('Load pre-computed cosine similarity matrix from {}'.format(args.counter_fitting_cos_sim_path))
        with open(args.counter_fitting_cos_sim_path, "rb") as fp:
            sim_lis = pickle.load(fp)
    else:
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

    stop_words_set = criteria.get_stopwords()
    print('Start attacking!')

    sims = []
    perts = []
    queries = []

    qrs_stopp=[100,200,300,400,500,600,700,800,900,1000]


    for idx, (text, true_label) in enumerate(data):
        hash_qrs = {}
        random_text, random_qrs, orig_label_, flag = random_attack(args.top_k_words, text_ls=text,
                                                                   true_label=true_label,
                                                                   predictor=predictor, word2idx=word2idx,
                                                                   idx2word=idx2word,
                                                                   cos_sim=sim_lis,

                                                                   sim_score_window=args.sim_score_window,
                                                                   batch_size=args.batch_size)

        if flag:
            orig_label = orig_label_
            print("=" * 10 + "开始优化" + "=" * 10)
            print("Attacked: " + str(idx))



            optim_step={}
            for kp in qrs_stopp:
                optim_step[kp] = []
            new_text, db_num_changed, random_changed, orig_label, \
            new_label, db_num_queries, db_sim, random_sim, db_qrs = attack(args.fuzz, optim_step,orig_label,
                                                                                args.top_k_words,
                                                                                args.allowed_qrs,
                                                                                idx, text[:], random_text[:],
                                                                                true_label, predictor,
                                                                                stop_words_set,
                                                                                word2idx, idx2word, sim_lis,
                                                                                sim_predictor=use,
                                                                                sim_score_threshold=args.sim_score_threshold,
                                                                                import_score_threshold=args.import_score_threshold,
                                                                                sim_score_window=args.sim_score_window,
                                                                                synonym_num=args.synonym_num,
                                                                                batch_size=args.batch_size,
                                                                                embed_func=args.counter_fitting_embeddings_path,
                                                                                n_sample=5,
                                                                                k_sample=5,
                                                                                k_threshold=0,
                                                                                qrs_limits=1000,
                                                                                qrs_stopp=qrs_stopp
                                                                                )

                
            sims.append(float(db_sim))
            perts.append(get_pert_rate(' '.join(text),new_text))
            queries.append(db_qrs)

            print(np.mean(sims),'\t',np.mean(perts),'\t',np.mean(queries))
            for k,v in optim_step.items():
                optim_step[k][0]= float(optim_step[k][0])
                optim_step[k][-1] = int(optim_step[k][-1])
                if not v[-1]:
                    exit(0)
            

            optim_step['text_ls'] = ' '.join(text)
            print(np.mean(sims),np.mean(perts))




            
if __name__ == "__main__":
    main()
