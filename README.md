Official codebase for AAAI 2026 Oral paper: Probabilistic Hash Embeddings for Online Learning of Categorical Features

---

To reproduce the experiments of PHE in the paper, please make sure the required software packages listed in `requirements.txt` are installed. To install the required packages:

```
pip install -r requirements.txt
```

## Online tabular data experiments

- Adult
  
  ```
  python main.py --dataset-name=adult_online --config-file=config_adult_online.yml --incre-col=education --reg-weight=1.0
  ```
* Bank
  
  ```
  python main.py --dataset-name=bank_online --config-file=config_bank_online.yml --incre-col=poutcome --reg-weight=1.0
  ```

* Mushroom
  
  ```
  python main.py --dataset-name=mushroom_online --config-file=config_mushroom_online.yml --incre-col=odor --reg-weight=1.0
  ```

* CoverType 
  
  *Note the original data is large. We only keep the top 50,000 records for demonstration.* Complete dataset is available here: https://archive.ics.uci.edu/dataset/31/covertype
  
  ```
  python main.py --dataset-name=covertype_online --config-file=config_covertype_online.yml --incre-col=wilderness_area --reg-weight=1.0
  ```

## Online sequence modeling experiments

* Retail
  
  *Note the original data is large. We only keep the top 50,000 records for demonstration.* Complete dataset is available here: https://archive.ics.uci.edu/dataset/352/online+retail
  
  ```
  python main.py --dataset-name=retail_online --config-file=config_retail_online.yml --incre-col=StockCode --reg-weight=1.0
  ```
  
## Online recommendation system experiments

* MovieLens-32M
  
  *Note the original data is large. We only keep the top 50,000 records for demonstration.* Complete dataset is available here: https://grouplens.org/datasets/movielens/32m/
  
  ```
  python main.py --dataset-name=movielens --config-file=config_movielens_mlp2.yml
  ```
