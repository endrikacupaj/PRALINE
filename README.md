# Contrastive Representation Learning for Conversational Question Answering over Knowledge Graphs

This paper addresses the task of conversational question answering (ConvQA) over knowledge graphs (KGs). The majority of existing ConvQA methods rely on full supervision signals with a strict assumption of the availability of gold logical forms of queries to extract answers from the KG. However, creating such a gold logical form is not viable for each potential question in a real-world scenario. Hence, in the case of missing gold logical forms, the existing information retrieval-based approaches use weak supervision via heuristics or reinforcement learning, formulating ConvQA as a KG path ranking problem. Despite missing gold logical forms, an abundance of conversational contexts, such as entire dialog history with fluent responses and domain information, can be incorporated to effectively reach the correct KG path. This work proposes a contrastive representation learning-based approach to rank KG paths effectively. Our approach solves two key challenges. Firstly, it allows weak supervision-based learning that omits the necessity of gold annotations. Second, it incorporates the conversational context (entire dialog history and domain information) to jointly learn its homogeneous representation with KG paths to improve contrastive representations for effective path ranking. We evaluate our approach on standard datasets for ConvQA, on which it significantly outperforms existing baselines on all domains and overall. Specifically, in some cases, the Mean Reciprocal Rank (MRR) and Hit@5 ranking metrics improve by absolute $10$ and $18$ points, respectively, compared to the state-of-the-art performance.

<img src="image/praline_architecture.png?raw=true" alt="PRALINE architecture" width="600"/>

PRALINE (**P**ath **R**anking for convers**A**tiona**L** quest**I**on a**N**sw**E**ring) architecture. It consists of three steps: 1) Extract KG paths and domains and represent them using a BERT model. 2) Learn the conversational context using a BART model and a domain identification pointer. 3) A contrastive ranking module that learns a joint embedding space $\phi^{c}, \phi^{p}$ for the conversation (contextual embeddings $h^{(enc)}$ \& selected domain embeddings $h^{(dm)}$) and the context path $h^{(p)}$.

## Requirements and Setup

Python version >= 3.7

PyTorch version >= 1.10.0

``` bash
# clone the repository
git clone https://github.com/endrikacupaj/PRALINE.git
cd PRALINE
pip install -r requirements.txt
```

## Train
For training you will need to adjust the paths in [args](args.py) file. At the same file you can also modify and experiment with different model settings.
``` bash
# train PRALINE
python train.py
```

## Test
For testing, first consider the [args](args.py) file for specifying the desired checkpoint path, after you can run the test file:
``` bash
# test PRALINE
python test.py
```

## License
The repository is under [MIT License](LICENSE).

## Cite
```bash
@inproceedings{10.1145/3511808.3557267,
  author = {Kacupaj, Endri and Singh, Kuldeep and Maleshkova, Maria and Lehmann, Jens},
  title = {Contrastive Representation Learning for Conversational Question Answering over Knowledge Graphs},
  year = {2022},
  isbn = {9781450392365},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3511808.3557267},
  doi = {10.1145/3511808.3557267},
  abstract = {This paper addresses the task of conversational question answering (ConvQA) over knowledge graphs (KGs). The majority of existing ConvQA methods rely on full supervision signals with a strict assumption of the availability of gold logical forms of queries to extract answers from the KG. However, creating such a gold logical form is not viable for each potential question in a real-world scenario. Hence, in the case of missing gold logical forms, the existing information retrieval-based approaches use weak supervision via heuristics or reinforcement learning, formulating ConvQA as a KG path ranking problem. Despite missing gold logical forms, an abundance of conversational contexts, such as entire dialog history with fluent responses and domain information, can be incorporated to effectively reach the correct KG path. This work proposes a contrastive representation learning-based approach to rank KG paths effectively. Our approach solves two key challenges. Firstly, it allows weak supervision-based learning that omits the necessity of gold annotations. Second, it incorporates the conversational context (entire dialog history and domain information) to jointly learn its homogeneous representation with KG paths to improve contrastive representations for effective path ranking. We evaluate our approach on standard datasets for ConvQA, on which it significantly outperforms existing baselines on all domains and overall. Specifically, in some cases, the Mean Reciprocal Rank (MRR) and Hit@5 ranking metrics improve by absolute 10 and 18 points, respectively, compared to the state-of-the-art performance.},
  booktitle = {Proceedings of the 31st ACM International Conference on Information &amp; Knowledge Management},
  pages = {925–934},
  numpages = {10},
  keywords = {conversations, question answering, contrastive learning, kg},
  location = {Atlanta, GA, USA},
  series = {CIKM '22}
}
```
