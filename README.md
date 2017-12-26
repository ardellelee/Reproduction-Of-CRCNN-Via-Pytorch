# The-Reproduction-Of-CRCNN-Via-Pytorch
I tried to reproduce [CRCNN](https://arxiv.org/abs/1504.06580) with pytorch for my future work.

This paper discoveried a new loss named pairwise ranking loss which is the only difference from CNN+softmax.

I have done some experiments in the choice of embedding.
The paper used word2vec to pretrain the embedding vector. So I attempted to use random embedding vector with 400 dim.
but I got 65% f1. Then I tried to use pretrianed vector with 50 dim. the f1 score increased to 70%. But it still can not be the same high like the paper's.

i'm the new in pytorch. thank you very much for any advice.
