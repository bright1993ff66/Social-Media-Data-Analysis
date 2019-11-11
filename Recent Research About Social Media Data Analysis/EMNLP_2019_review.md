#  EMNLP 2019 Review

## 1. Overview

The [EMNLP 2019]( https://www.emnlp-ijcnlp2019.org/ ) started from November 3 to November 7. The proceedings could be checked in [here]( https://github.com/roomylee/EMNLP-2019-Papers ). After attending the workshops, tutorials and main conference, some trends could be captured:

- [Attention Mechanism]( https://blog.floydhub.com/attention-mechanism/ ) is still one of the most discussed topics in NLP community, since it has been widely applied in many fields from machine translation to question answering
- Some pretrained models such as [ELMo]( https://allennlp.org/elmo ), [Bert]( https://towardsdatascience.com/bert-for-dummies-step-by-step-tutorial-fb90890ffe03 ) has also been applied in many different studies
- [Graph model]( https://en.wikipedia.org/wiki/Graphical_model ) becomes a hot topic this year, since it could encode complex structures and global information. This module has been applied in many fields, including text classification, social media network construction, etc.
- Lastly, some works combine the text and the images together. Some interesting works would be listed in this summary

## 2. Highlights

In this section, the in-depth description of each bullet point listed in the **Overview** and the corresponding concrete examples are given in this section.

### 2.1 Attention Mechanism

To be continued...

### 2.2 Pretrained Models

To be continued...

### 2.3 Graph Model & Applications

A graph model is composed of **node** and **edge**. By defining node and edge in different scenarios, combing with attention, we could finish some real-world downstream tasks. For instance, 

- In **text classification**, we could let each node to represent one word. More specifically, The initial representation of each node could be generated using pretrained word embedding like [FastText]( https://fasttext.cc/ ). In the following training process, the representation of each node would be updated. Then the edge information between words could be computed using popular metrics like [Pointwise Mutual Information]( https://en.wikipedia.org/wiki/Pointwise_mutual_information ). For instance, for a word pair $(i,j)$, The PMI metric could be calculated as the following:
  
  ![pmi]( https://github.com/bright1993ff66/Image_Text_Sentiment/blob/master/figures/pmi.PNG )
  
  Where:
  
  ![pij]( https://github.com/bright1993ff66/Image_Text_Sentiment/blob/master/figures/pmi_pij.PNG )
  
  and:
  
  ![pmi_pi]( https://github.com/bright1993ff66/Image_Text_Sentiment/blob/master/figures/pmi_pi.PNG )
  
  The $\# W(i)$ represents the number of sliding windows in a corpus that contain word $i$, $\#W(i, j)$ is the number of sliding windows that contain both word $i$ and $j$, and $\#W$ is the total number of sliding windows in the corpus. A positive PMI value implies a high semantic correlation of words in a corpus.

  Some representative works are:

  - [Graph Convolutional Networks for Text Classification]( https://arxiv.org/abs/1809.05679 )
  - [Text Level Graph Neural Network for Text Classification]( https://arxiv.org/abs/1910.02356 )

- In **social media user modelling**, we use each node to represent one user. The node could be represented by, for instance, author representation using [Node2Vec]( https://cs.stanford.edu/~jure/pubs/node2vec-kdd16.pdf ). The edge weight could be computed using the [Graph Attention Networks]( https://arxiv.org/pdf/1710.10903.pdf ). Some representative papers are:

  - [You Shall Know a User by the Company It Keeps: Dynamic Representations for Social Media Users in NLP]( https://arxiv.org/abs/1909.00412 )
  - [ Learning Invariant Representations of Social Media Users ]( https://www.aclweb.org/anthology/D19-1178.pdf )

### 2.4 Image & Text

Some subfields which lie in the intersection of NLP & CV such as [Image Captioning]( https://www.tensorflow.org/tutorials/text/image_captioning ) still receives a lot of attention this year. It is mainly because of the following reasons:

- Images produce additional information in the downstream NLP tasks. For instance, in machine translation, some model could make use of the visual context to guide the translation decoder. You could check [Distilling Translations with Visual Awareness]( https://arxiv.org/pdf/1906.07701v1.pdf )
- The relationship between the image-text pair is still not clear. Some works such as [LXMERT]( https://arxiv.org/pdf/1908.07490.pdf ) uses the [transformer]( https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf ) to encode both objects in image and texts. By applying to five diverse representative learning tasks, the model reach the highest performance in the visual-reasoning tasks. Other works, such as [ Integrating Text and Image: Determining Multimodal Document Intent in Instagram Posts ]( https://arxiv.org/pdf/1904.09073.pdf ), build the image-text pair dataset by human annotation and use deep neural net to automatically predict the relationship between image and the relevant sentence

## 3. Future Research Paths

In the future, 

