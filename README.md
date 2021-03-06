One day with [MeLi Data
Challenge 2019](https://ml-challenge.mercadolibre.com/)
================

In less than 8 hours of work, I am going to work as much as possible in
the [MeLi Data Challenge 2019](https://ml-challenge.mercadolibre.com/).

Data description:
[Link](https://ml-challenge.mercadolibre.com/downloads).

## Getting the data

``` bash
# Clone current repository
git clone https://github.com/jcrodriguez1989/MeLi_Data_Challenge_2019.git
cd MeLi_Data_Challenge_2019

# Download MeLi Challenge input data
wget https://meli-data-challenge.s3.amazonaws.com/train.csv.gz -P data/
gunzip data/train.csv.gz # uncompress it
wget https://meli-data-challenge.s3.amazonaws.com/test.csv -P data/
```

## Download [fasttext pre-trained embedding models](https://fasttext.cc/docs/en/crawl-vectors.html)

``` bash
wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.es.300.bin.gz -P models/
wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.pt.300.bin.gz -P models/

gunzip models/cc.es.300.bin.gz
gunzip models/cc.pt.300.bin.gz
```

## Data analysis

Follow R scripts in the file prefix (`"XX_"`) order.
