from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from nltk.tokenize import RegexpTokenizer
from collections import defaultdict
from miscc.config import cfg

import torch
import torch.utils.data as data
from torch.autograd import Variable
import torchvision.transforms as transforms

import os
import sys
import h5py
import numpy as np
import pandas as pd
from PIL import Image

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle


class TextDataset(data.Dataset):
    def __init__(self, data_dir, split='train', base_size=64,
                 transform=None, target_transform=None):

        self.transform = transform
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.target_transform = target_transform
        self.embeddings_num = cfg.TEXT.CAPTIONS_PER_IMAGE

        self.imsize = []
        for i in range(cfg.TREE.BRANCH_NUM):
            self.imsize.append(base_size)
            base_size = base_size * 2

        self.data = []
        self.data_dir = data_dir
        self.bbox = None

        # parse caption from .h5 file and save as pickle file
        train_h5_filename = os.path.join(data_dir, 'fashiongen_256_256_train.h5')
        test_h5_filename = os.path.join(data_dir, 'fashiongen_256_256_validation.h5')

        # open the h5 files
        self.train_file_h5 = h5py.File(train_h5_filename, mode='r')
        self.test_file_h5 = h5py.File(test_h5_filename, mode='r')

        self.current_h5_file = self.train_file_h5 if split == 'train' else self.test_file_h5

        # filter based on selected categories
        selected_categories = ['TOPS', 'SWEATERS', 'PANTS', 'JEANS', 'SHIRTS', 'DRESSES', 'SHORTS', 'SKIRTS']
        self.filtered_image_indexes = self.filter_by_categories(selected_categories, split)

        self.captions, self.ixtoword, self.wordtoix, self.n_words = self.load_text_data(data_dir, selected_categories, split)

        self.number_example = len(self.filtered_image_indexes)
        self.class_id = self.load_class_id(self.number_example)

        # keep open only the specified split.
        # Both (train and test) has been opened to create the dictionary.
        if split == 'train':
            self.test_file_h5.close()
            self.test_file_h5 = None
        else:
            self.train_file_h5.close()
            self.train_file_h5 = None

    def filter_by_categories(self, selected_categories, split):
        # select train or test dataset h5 file
        file_h5 = self.train_file_h5 if split == 'train' else self.test_file_h5

        indexes = []
        for i in range(len(file_h5['input_category'])):
            if file_h5['input_category'][i] in selected_categories:
                indexes.append(i)
        return indexes

    def load_captions(self, split, selected_categories):
        # select train or test dataset h5 file
        file_h5 = self.train_file_h5 if split == 'train' else self.test_file_h5

        word_count_overflow = 0  # number of caption that exceed the max word count
        caption_count = len(file_h5['input_description'])
        print('loading %d captions (%s)' % (caption_count, split))

        all_captions = []
        for i in range(caption_count):
            if file_h5['input_category'][i] in selected_categories:
                # get caption for this sample
                caption = str(file_h5['input_description'][i])

                # keep first sentence only
                index = caption.find('.')
                if index is not -1:
                    caption = caption[:index]

                if i % 1000 == 0:
                    print('%d / %d' % (i, caption_count))

                if len(caption) == 0:
                    raise ValueError('Very empty caption at index %d of dataset %s' % (i, split))

                caption = caption.replace("\ufffd\ufffd", " ")

                # picks out sequences of alphanumeric characters as tokens and drops everything else
                tokenizer = RegexpTokenizer(r'\w+')
                tokens = tokenizer.tokenize(caption.lower())
                #print('tokens', tokens)

                if len(tokens) == 0:
                    raise ValueError('Empty caption at index %d of dataset %s' % (i, split))

                tokens_new = []
                for t in tokens:
                    t = t.encode('ascii', 'ignore').decode('ascii')
                    if len(t) > 0:
                        tokens_new.append(t)

                all_captions.append(tokens_new)

                if len(tokens_new) > cfg.TEXT.WORDS_NUM:
                    word_count_overflow += 1

        print('%d captions exceeding the max word count of %d' % (word_count_overflow,  cfg.TEXT.WORDS_NUM))

        return all_captions

    def build_dictionary(self, train_captions, test_captions):
        word_counts = defaultdict(float)
        captions = train_captions + test_captions
        for sent in captions:
            for word in sent:
                word_counts[word] += 1

        vocab = [w for w in word_counts if word_counts[w] >= 0]

        ixtoword = {}
        ixtoword[0] = '<end>'
        wordtoix = {}
        wordtoix['<end>'] = 0
        ix = 1
        for w in vocab:
            wordtoix[w] = ix
            ixtoword[ix] = w
            ix += 1

        train_captions_new = []
        for t in train_captions:
            rev = []
            for w in t:
                if w in wordtoix:
                    rev.append(wordtoix[w])
            # rev.append(0)  # do not need '<end>' token
            train_captions_new.append(rev)

        test_captions_new = []
        for t in test_captions:
            rev = []
            for w in t:
                if w in wordtoix:
                    rev.append(wordtoix[w])
            # rev.append(0)  # do not need '<end>' token
            test_captions_new.append(rev)

        return [train_captions_new, test_captions_new,
                ixtoword, wordtoix, len(ixtoword)]

    def load_text_data(self, data_dir, selected_categories, split):
        filepath = os.path.join(data_dir, 'captions2_attngan.pickle')

        if not os.path.isfile(filepath):
            train_captions = self.load_captions('train', selected_categories)
            test_captions = self.load_captions('test', selected_categories)

            train_captions, test_captions, ixtoword, wordtoix, n_words = \
                self.build_dictionary(train_captions, test_captions)

            # save in pickle file
            with open(filepath, 'wb') as f:
                pickle.dump([train_captions, test_captions, ixtoword, wordtoix], f, protocol=2)
                print('Save to: ', filepath)
        else:
            # load preparsed pickle file
            with open(filepath, 'rb') as f:
                x = pickle.load(f)
                train_captions, test_captions = x[0], x[1]
                ixtoword, wordtoix = x[2], x[3]
                del x
                n_words = len(ixtoword)
                print('Load from: ', filepath)

        if split == 'train':
            # a list of list: each list contains the indices of words in a sentence
            captions = train_captions
        else:  # split=='test'
            captions = test_captions

        return captions, ixtoword, wordtoix, n_words

    def load_class_id(self, total_num):
        class_id = np.arange(total_num)
        return class_id

    def get_caption(self, index):
        # a list of indices for a sentence
        sent_caption = np.asarray(self.captions[index]).astype('int64')

        if (sent_caption == 0).sum() > 0:
            print('ERROR: do not need END (0) token', sent_caption)

        num_words = len(sent_caption)

        # pad with 0s (i.e., '<end>')
        x = np.zeros((cfg.TEXT.WORDS_NUM, 1), dtype='int64')
        x_len = num_words
        if num_words <= cfg.TEXT.WORDS_NUM:
            x[:num_words, 0] = sent_caption
        else:
            ix = list(np.arange(num_words))  # 1, 2, 3,..., maxNum
            np.random.shuffle(ix)
            ix = ix[:cfg.TEXT.WORDS_NUM]
            ix = np.sort(ix)
            x[:, 0] = sent_caption[ix]
            x_len = cfg.TEXT.WORDS_NUM

        return x, x_len

    def get_imgs(self, index):
        h5_index = self.filtered_image_indexes[index]
        img = Image.fromarray(self.current_h5_file['input_image'][h5_index].astype('uint8'), 'RGB')

        if self.transform is not None:
            img = self.transform(img)

        ret = []
        if cfg.GAN.B_DCGAN:
            ret = [self.norm(img)]
        else:
            for i in range(cfg.TREE.BRANCH_NUM):
                # print(imsize[i])
                if i < (cfg.TREE.BRANCH_NUM - 1):
                    re_img = transforms.Scale(self.imsize[i])(img)
                else:
                    re_img = img
                ret.append(self.norm(re_img))

        return ret

    def __getitem__(self, index):
        key = index
        cls_id = self.class_id[index]

        imgs = self.get_imgs(index)

        caps, cap_len = self.get_caption(index)

        return imgs, caps, cap_len, cls_id, key

    def __len__(self):
        return len(self.filtered_image_indexes)
