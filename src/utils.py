from datasets import load_dataset, Dataset
import pandas as pd
import numpy as np
import os
import re

from rouge import Rouge
from copy import deepcopy
import emoji
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from IPython.display import clear_output
import language_tool_python
from sklearn.metrics import pairwise_distances

from better_profanity import profanity
custom_badwords = ['garbage', 'junk', 'dumb', 'crappy', 'crappiest', 'crappier', 'sucks', 'sucker', 'sucky']
profanity.add_censor_words(custom_badwords)

import torch
import tensorflow as tf
import tensorflow_hub as hub

tool = language_tool_python.LanguageTool('en-US')
module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
embed = hub.load(module_url)
rouge = Rouge()
analyzer = SentimentIntensityAnalyzer()

to_remove = ["&#34", "&quot", "<br />", "*", "/", "@", '\\', "#", "%", "^", "&", "~", "'", '"', '-', '—', '(', ')']
punkt = ['.', '?', ';', ':', '!', ',']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

oos_cat = 'Arts_Crafts_and_Sewing'

def process(text, remove, lower=True):
    text = text.split("[[VIDEOID")[-1].split("]] ")[-1]
    if lower: text = text.lower()
    for r in remove: 
        text = text.replace(r, ' ')
        text = text.replace('  ', ' ')
    return text

def filtering(title, review, rating, timestamp, name, category, exclude=[]):
    L = len(rating)
    count = 0
    final = []
    for idx, (t_orig, r_orig, rat, time, n_orig, c) in enumerate(zip(title, review, rating, timestamp, name, category)): 
        if len(t_orig) >= 15 and len(t_orig) <= 80 and len(r_orig) >= 200 and len(r_orig) <= 1500:
            t = process(t_orig, to_remove + punkt)
            r_proc = process(r_orig, to_remove + punkt)
            n = process(n_orig, to_remove + punkt)
            if (not t[:15] in r_proc) and (not t[-15:] in r_proc):
                r = sent_tokenize(process(r_proc, to_remove))
                rouge_s = [rouge.get_scores(t, process(sent_r, punkt))[0]['rouge-1']['f'] for sent_r in r]
                if np.mean(rouge_s) > 0.125:
                    review_title_sim = similarity(r_proc, t)
                    review_product_sim =  similarity(r_proc, n)
                    if review_title_sim > 0.2 and review_product_sim < 0.6 and review_product_sim > 0.1:
                        if not n_orig in exclude:
                            pred_sentiment = analyzer.polarity_scores(t)['compound']
                            real_sentiment = (rat - 3.0)/2.0
                            if (pred_sentiment == 0.0 and np.abs(real_sentiment) == 1.0) \
                            or (pred_sentiment < -0.01 and real_sentiment > 0.01) \
                            or (pred_sentiment > 0.01 and real_sentiment < -0.01):
                                pass
                            else:
                                t_words = word_tokenize(t)
                                if len(t_words) > 3 and not profanity.contains_profanity(t):
                                    final.append([t_orig, r_orig, rat, time, n_orig, c])
                                    count += 1
        if (idx+1) % 100 == 0 or L-idx < 100:
            clear_output(wait=True)
            print("Converted data point {}/{}. Number of valid review-title pairs: {}".format(idx+1, L, count))
    return final

def filter_by_ratings(data):
    new_data = deepcopy(data)
    num_r = [len(data[data['rating']==r]) for r in range(1,6)]
    for i in range(len(num_r)):
        for _ in range(num_r[i] - np.min(num_r)):
            new_data.drop(new_data[new_data['rating']==i+1].index[-1], inplace=True)
    return new_data

def to_prompt(pipeline, title, review, instruction=None, rating=None, product_name=None, extra_inst=True):
    
    extra_inst_message = " Your only output should be the title itself. Do not mention the user rating in the title."
    
    if instruction is None:
        instruction = "Generate the best succinct title for the following product review." + \
        (extra_inst_message if extra_inst else "") + \
        ("" if rating is None else " Product rating: {}/5 stars.".format(int(rating))) + \
        ("" if product_name is None else " Product categories: '{}'.".format(product_name))
    
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": review}
        ]
        if not title == "":
            messages.append(
                {"role": "assistant", "content": '"'+title+'"'}
            )
        prompt = pipeline.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
        )
    
    return prompt

def convert_data(pipeline, data, use_rating=True, use_product_name=True, include_answer=True):
    new_data = []
    for idx, d in data.iterrows():
        new_data.append(
            to_prompt(
                pipeline,
                d['title'] if include_answer else "", 
                d['text'], 
                rating=d['rating'] if use_rating else None,
                product_name=d['category'] if use_product_name else None
            )
        )
    return new_data

def similarity(r, t):
    re = embed([r])
    te = embed([t])
    return tf.tensordot(re, te, 2)/(tf.norm(re)*tf.norm(te)) # cosine similarity
    # return 1 - np.arccos(tf.tensordot(re, te, 2)/(tf.norm(re)*tf.norm(te)))*(1/np.pi) # AES
    
def duplicate(dataset_asins, meta_asins, names, categories):
    new_names = []
    new_cats = []
    
    j = 0
    L = len(dataset_asins)
    for i, (asin, name, cat) in enumerate(zip(meta_asins, names, categories)):
        while j < L and asin == dataset_asins[j]:
            new_names.append(name)
            new_cats.append(cat)
            j += 1
    return new_names, new_cats 

def load_prompts(pipeline, cat=None):
    if cat is None:
        train_even = pd.read_csv('train_combined.csv')
        valid_even = pd.read_csv('valid_combined.csv')
        test_even = pd.read_csv('test_combined.csv')
    else:
        train_even = pd.read_csv(os.path.join(cat, 'train_even.csv'))
        valid_even = pd.read_csv(os.path.join(cat, 'valid_even.csv'))
        test_even = pd.read_csv(os.path.join(cat, 'test_even.csv'))

    prompts_train = Dataset.from_dict(
        {'text': convert_data(pipeline, train_even)}
    )
    prompts_valid = Dataset.from_dict(
        {'text': convert_data(pipeline, valid_even, include_answer=False)}
    )
    prompts_test = Dataset.from_dict(
        {'text': convert_data(pipeline, test_even, include_answer=False)}
    )
    
    return prompts_train, prompts_valid, prompts_test

def metrics(titles, reviews, ratings):
    L = len(ratings)
    # Title length, number of words, uniqueness, rouge-1, rouge-2, rouge-L, 
    # similarity, sentiment, rating, profanity, special chars
    final = np.zeros((L, 13))
    for i, (to, ro, rat) in enumerate(zip(titles, reviews, ratings)): 
        t = process(to, to_remove + punkt)
        r = process(ro, to_remove + punkt)
        tw = word_tokenize(t)
        final[i, 0] = len(t)
        final[i, 1] = len(tw)
        
        if len(tw) > 2:
            final[i, 2] = ((tw[0]+' '+tw[1]+' '+tw[2] in r) or (tw[-3]+' '+tw[-2]+' '+tw[-1] in r))
        
        rs = sent_tokenize(process(ro, to_remove))
        rouge_1 = []
        rouge_2 = []
        rouge_L = []
        sim = []
        for sent in rs:
            sent = process(sent, punkt)
            try: 
                rouge_all = rouge.get_scores(t, sent)[0]
                rouge_1.append(rouge_all['rouge-1']['r'])
                rouge_2.append(rouge_all['rouge-2']['r'])
                rouge_L.append(rouge_all['rouge-l']['r'])
                sim.append(similarity(t, sent))
            except ValueError:
                rouge_1.append(0)
                rouge_2.append(0)
                rouge_L.append(0)  
                sim.append(0)
        final[i, 3] = np.max(rouge_1)
        final[i, 4] = np.max(rouge_2)
        final[i, 5] = np.max(rouge_L)
        final[i, 6] = np.max(sim)
        
        final[i, 7] = analyzer.polarity_scores(t)['compound']
        final[i, 8] = (rat - 3.0)/2.0
        final[i, 9] = profanity.contains_profanity(t)
        final[i, 10] = len(re.sub('[^\^&*$.,!?@#%()�]+' ,'', to))/len(to)
        final[i, 11] = len(emoji.emoji_list(to))
        final[i, 12] = to.isupper()
    return final

# MMD
# Code adapted with permission from https://github.com/IDEALLab/CEBGAN_JMD_2021/blob/main/CEBGAN/src/utils/metrics.py 
def gaussian_kernel(X, Y, sigma=2.0):
    beta = 1. / (2. * sigma**2)
    dist = pairwise_distances(X, Y)
    s = beta * dist.flatten()
    return np.exp(-s)

def MMD(X_gen, X_test):
    X_gen = X_gen.reshape((X_gen.shape[0], -1))
    X_test = X_test.reshape((X_test.shape[0], -1)) 
    mmd = np.mean(gaussian_kernel(X_gen, X_gen)) - \
        2 * np.mean(gaussian_kernel(X_gen, X_test)) + \
        np.mean(gaussian_kernel(X_test, X_test))          
    return np.sqrt(mmd)

# R-Div
# Code adapted with permission from https://github.com/IDEALLab/CEBGAN_JMD_2021/blob/main/CEBGAN/src/utils/metrics.py 
def variance(X):
    cov = np.cov(X.T)
    var = np.trace(cov)/cov.shape[0]
    return var

def rdiv(X_train, X_gen):
    X_train = np.squeeze(X_train)
    X_train = X_train.reshape((X_train.shape[0], -1))
    train_div = variance(X_train)
    X_gen = X_gen.reshape((X_gen.shape[0], -1))
    gen_div = variance(X_gen)
    rdiv = gen_div/train_div
    return rdiv