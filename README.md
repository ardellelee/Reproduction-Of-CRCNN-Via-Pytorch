# The-Reproduction-Of-CRCNN-Via-Pytorch
I tried to reproduce [CRCNN](https://arxiv.org/abs/1504.06580) with pytorch for my future work.

This paper discoveried a new loss named pairwise ranking loss which is the only difference from CNN+softmax.

I have done some experiments in the choice of embedding.
The paper used word2vec to pretrain the embedding vector. So I attempted to use random embedding vector with 400 dim.
but I got 65% f1. Then I tried to use pretrianed vector with 50 dim. the f1 score increased to 70%. But it still can not be the same high like the paper's.

the best f1 score is:

epoch: 46 f1: 76.9 %   loss: 4.02725402832 test_f1: 69.9360198625 %

epoch: 47 f1: 76.75 %   loss: 4.01846038818 test_f1: 69.1668258212 %

epoch: 48 f1: 77.9 %   loss: 3.95451751709 test_f1: 70.0405844156 %

epoch: 49 f1: 78.325 %   loss: 3.88577056885 test_f1: 69.7450343774 %

epoch: 50 f1: 78.15 %   loss: 3.90136352539 test_f1: 69.9598930481 %

i'm new in pytorch. thank you very much for any advice.
