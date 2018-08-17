#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#如果使用sigmoid激活函数，则交叉熵损失函数一般肯定比均方差损失函数好
#如果是DNN用于分类，则一般在输出层使用softmax激活函数和对数似然损失函数
#ReLU激活函数对梯度消失问题有一定程度的解决，尤其是在CNN模型中

import numpy as np
from hmmlearn import hmm
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

states = ["box 1", "box 2", "box3"]
n_states = len(states)

observations = ["red", "white"]
n_observations = len(observations)

start_probability = np.array([0.2, 0.4, 0.4])

transition_probability = np.array([
  [0.5, 0.2, 0.3],
  [0.3, 0.5, 0.2],
  [0.2, 0.3, 0.5]
])

emission_probability = np.array([
  [0.5, 0.5],
  [0.4, 0.6],
  [0.7, 0.3]
])

model = hmm.MultinomialHMM(n_components=n_states)
model.startprob_=start_probability
model.transmat_=transition_probability
model.emissionprob_=emission_probability

seen = np.array([[0,1,0]]).T
logprob, box = model.decode(seen, algorithm="viterbi")
print("The ball picked:", ", ".join(map(lambda x: observations[x], seen.flatten())))
print("The hidden box", ", ".join(map(lambda x: states[x], box.flatten())))

box2 = model.predict(seen)
print("The ball picked:", ", ".join(map(lambda x: observations[x], seen.flatten())))
print("The hidden box", ", ".join(map(lambda x: states[x], box2.flatten())))

#score函数返回的是以自然对数为底的对数概率值，ln0.13022≈−2.0385
print(model.score(seen))
