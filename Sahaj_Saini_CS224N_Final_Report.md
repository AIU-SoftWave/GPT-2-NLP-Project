Sentiment Classification using GPT-2 Representations

Stanford CS224N Default Project

 Sahaj Saini

 Stanford University

 sahaj08@stanford.edu

 March 15, 2026

Abstract

Pretrained language models have dramatically improved performance across many natural language processing tasks. In this project, we evaluate the effectiveness of using pretrained GPT-2 representations for sentiment classification. We experiment with two strategies: using GPT-2 as a frozen feature extractor with a linear classification head, and fine-tuning the entire model on the downstream task. Experiments are conducted on two sentiment classification datasets: the Stanford Sentiment Treebank(SST) and the CFIMDB movie review dataset. Our experiments show that pretrained GPT-2 embeddings already provide strong baseline performance,while fine-tuning the full model further improves accuracy. These results demonstrate that large pretrained language models transfer effectively to downstream classification tasks.

1 Key Information to include

•External Collaborators: None

.Sharing project: No

2 Introduction

 Recent advances in natural language processing have been driven by large pretrained language models. GPT-2[1] is a transformer-based language model.Models such as GPT-2 learn general language representations from massive corpora through self-supervised objectives. These pretrained models can then be adapted to downstream tasks such as sentiment classification, question answering,and summarization.


Sentiment analysis is a fundamental NLP task that involves identifying the emotional polarity expressed in text. Applications include analyzing product reviews, social media posts, and customer feedback. Earlier approaches relied on manually engineered features or traditional machine learning algorithms. Deep neural networks later improved performance by learning representations directly from text data.

Pretrained transformer models have further improved results by learning contextual representations of language. Instead of training models from scratch for each task, pretrained models can be reused as general feature extractors.Alternatively, they can be fine-tuned on a specific dataset.

In this project, we explore how pretrained GPT-2 representations can be used for sentiment classification. We compare two approaches: freezing the GPT-2 encoder and training only a classifier head, and fine-tuning the full GPT-2 model.We evaluate both approaches on two datasets: the Stanford Sentiment Treebank(SST) and the CFIMDB movie review dataset.

3 Related Work

Pretrained language models have become the dominant paradigm in modern NLP. Early work such as Word2Vec[2] and GloVe[3] introduced distributed word representations learned from large text corpora. While these embeddings capture semantic similarity, they produce static representations that do not depend on context.

Transformer architectures[4] introduced self-attention mechanisms that en-able models to capture long-range dependencies between words. These architec-tures form the foundation of most modern language models.

GPT-2[1] is a large autoregressive transformer trained to predict the next token in a sequence using a language modeling objective. Because GPT-2 is trained on large corpora, it learns general-purpose language representations that can transfer effectively to downstream tasks.

Another influential model, BERT[5], demonstrated the effectiveness of fine-tuning pretrained transformer models on a wide variety of NLP tasks. BERT introduced bidirectional contextual representations and achieved state-of-the-art results on multiple benchmarks.

Prior work has explored whether pretrained models should be used as frozen feature extractors or fully fine-tuned for downstream tasks. While frozen rep-resentations provide strong baselines, fine-tuning typically leads to improved performance by allowing the model to adapt to task-specific characteristics.

4 Approach

Our approach uses a pretrained GPT-2 model to encode sentences and pro-duce contextual token representations. These representations are then used for sentiment classification.


4.1 Model Architecture

Given an input sentence consisting of tokens  $x_{1}, x_{2},\ldots, x_{T}$  , the GPT-2 model produces contextual hidden representations:

$$
h_{1}, h_{2},\ldots, h_{T}
$$

where  $h_{t}$  represents the hidden state corresponding to token t.

Following common practice for GPT-style models, we use the representation of the final token  $h_{T}$  as the representation of the entire sentence. This vector is passed through a dropout layer followed by a linear classifier:

$$
y=W h_{T}+b
$$

where W and b are the parameters of the classification head.

4.2 Training Strategies

We evaluate two training strategies.

Last-Linear-Layer Training

In this configuration, all GPT-2 parameters are frozen and only the final classification layer is trained. This approach treats GPT-2 as a fixed feature extractor.

Full Model Fine-Tuning

In this configuration, all GPT-2 parameters are updated during training.Fine-tuning allows the model to adapt its internal representations to better capture sentiment information.

5 Experiments

5.1 Data

We evaluate our models on two sentiment classification datasets.

Stanford Sentiment Treebank(SST)

The SST dataset contains movie review sentences labeled with five sentiment classes: negative, somewhat negative, neutral, somewhat positive, and positive.

Dataset splits:

- Train: 8,544 examples

- Dev: 1,101 examples

- Test: 2,210 examples

CFIMDB

 The CFIMDB dataset consists of movie reviews labeled with binary sentiment(positive or negative).

Dataset splits:


.Train: 1,701 examples

.Dev: 245 examples

.Test: 488 examples

5.2 Evaluation Method

 Model performance is evaluated using classification accuracy on the development and test sets. Accuracy measures the proportion of correctly predicted sentiment labels.

5.3 Experimental Details

All experiments use a pretrained GPT-2 model as the encoder. The final token representation is used as the sentence embedding for classification.

Training uses the AdamW optimizer. Key hyperparameters include batch size, learning rate, dropout probability, and number of training epochs. Models are trained for up to ten epochs and the best checkpoint is selected based on development set accuracy.

5.4 Results



| Model | SST Dev Accuracy | CFIMDB Dev Accuracy |
| --- | --- | --- |
| Last Linear Layer | 0.451 | 0.829 |
| Full Model Fine-Tuning | 0.518 | 0.976 |


Table 1: Sentiment classification performance using GPT-2 representations.

Training only the linear classifier on top of frozen GPT-2 embeddings achieves 45.1% accuracy on the SST development set and 82.9% accuracy on the CFIMDB development set. Fine-tuning the entire GPT-2 model improves accuracy to $51.8\%$  on SST and  $97.6\%$  on CFIMDB.

These results demonstrate that pretrained GPT-2 representations capture useful semantic information for sentiment classification tasks.

6 Analysis

Fine-tuning the full GPT-2 model leads to noticeable improvements compared to using frozen representations. On SST, accuracy increases from 45.1% to 51.8%. The improvement is even larger on the CFIMDB dataset, where accuracy increases from 82.9% to 97.6%.

The difference between the datasets highlights the effect of task difficulty.SST is a five-class classification problem containing subtle sentiment distinctions,which makes it more challenging. In contrast, CFIMDB is a binary classification


task with longer reviews that often contain stronger sentiment cues, allowing the model to achieve higher accuracy.

These findings suggest that pretrained language models already encode useful sentiment information, while fine-tuning allows the model to adapt its internal representations to the target task.

7 Conclusion

In this project, we investigated the use of pretrained GPT-2 representations for sentiment classification. We compared two approaches: training a linear classifier on frozen GPT-2 embeddings and fine-tuning the entire GPT-2 model.

Our experiments show that frozen GPT-2 embeddings provide strong baseline performance on both datasets. Fine-tuning the full model further improves performance, demonstrating the benefit of adapting pretrained representations to the downstream task.

These results highlight the flexibility of pretrained language models and their ability to transfer knowledge across NLP tasks. Future work could explore larger language models, alternative classification architectures, or additional sentiment datasets.

References

[1] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language models are unsupervised multitask learners. OpenAI Technical Report, 2019.

[2] Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, and Jeffrey Dean.Distributed representations of words and phrases and their compositionality.In NeurIPS, 2013.

[3] Jeffrey Pennington, Richard Socher, and Christopher Manning. Glove: Global vectors for word representation. In EMNLP, 2014.

[4] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,Aidan Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need.In NeurIPS, 2017.

[5] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert:Pre-training of deep bidirectional transformers for language understanding.In NAACL, 2019.


