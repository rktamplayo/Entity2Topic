# Entity2Topic
Entity Commonsense Representation for Neural Abstractive Summarization

This code was used in the experiments of the research paper

**Reinald Kim Amplayo**<sup>\*</sup>, Seonjae Lim<sup>\*</sup> and Seung-won Hwang. **Entity Commonsense Representation for Neural Abstractive Summarization**. _NAACL_, 2018.

\* Authors have equal contributions

You will need the following data saved in a separate `data` folder:
- `word_vecs.txt`: word vectors (we used GloVe vectors which can be downloaded here: http://nlp.stanford.edu/data/glove.840B.300d.zip)
- `entity_vecs.txt`: entity vectors (we used wiki2vec vectors which can be downloaded here: https://github.com/idio/wiki2vec/raw/master/torrents/enwiki-gensim-word2vec-1000-nostem-10cbow.torrent)
- `train.article.txt` and `valid.article.txt`: contains the text to be summarized
- `train.title.txt` and `valid.title.txt`: contains the summarized text
- `train.entity.txt` and `valid.entity.txt`: contains the entities tagged using the format specified by wiki2vec here: https://github.com/idio/wiki2vec)

To run the code, several parameters are needed to be set in the `src/summarization.py`. Refer to our paper to determine the recommended values.

To train the model, execute the following code:

`python script/train.py`

Similarly, to test the model, execute the following code (although test will automatically come after training is done):

`python script/test.py`

To cite the paper/code, please use this BibTex:

```
@inproceedings{amplayo2017entity,
	Author = {Reinald Kim Amplayo and Seonjae Lim and Seung-won Hwang},
	Booktitle = {NAACL},
	Location = {New Orleans, LA},
	Year = {2018},
	Title = {Entity Commonsense Representation for Neural Abstractive Summarization},
}
```

If you have questions, send me an email: rktamplayo at yonsei dot ac dot kr