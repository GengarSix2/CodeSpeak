# CodeSpeak
Online repository of the paper submitted to ASE 2024 titled "CodeSpeak: Advancing Smart Contract Vulnerability Detection via LLM-Generated and Human Expert Knowledge".

# Dataset

We conduct experiments on the dataset from the paper below.

```
@inproceedings{10.1145/3543507.3583367,
author = {Qian, Peng and Liu, Zhenguang and Yin, Yifang and He, Qinming},
title = {Cross-Modality Mutual Learning for Enhancing Smart Contract Vulnerability Detection on Bytecode},
year = {2023},
isbn = {9781450394161},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
booktitle = {Proceedings of the ACM Web Conference 2023},
pages = {2220–2229},
numpages = {10},
location = {Austin, TX, USA},
series = {WWW '23}
}
```

## Dataset Format

We process the dataset into a JSONL format file, where each line represents a smart contract with the following structure.

- `"contract":` Processed smart contract source code.
- `"vulnerability_type":` Type of smart contract vulnerability to be detected.
- `"label":` Label.
- `"idx":` ID.

# Dependency

* python version: python 3.8.18
* numpy: 1.24.4
* pandas: 2.0.3
* Pillow: 9.3.0
* scikit-learn: 1.3.2
* torch: 2.1.0+cu118
* tqdm: 4.66.1
* transformers: 4.34.0
* tree-sitter: 0.20.2

# Run

Install the Python environment according to Dependency.

You can download the codebert model from hugging face in advance and put it in the models/codebert-base folder.

Run the generate_dataset.py file to generate the training set and test set.

```
python generate_dataset.py
```

If you want to add LLM-generated vulnerability descriptions to the training set, run the generate_dataset_description.py file to generate the training set and test set.

```
python data_augmentation_description.py
```

You can run the following command to perform a single training. After the training is completed, the results will be recorded in the saved-data folder in JSONL format.

- num_train_epochs &  train_batch_size & valid_batch_size & test_batch_size: Select the training parameters you want.
- data_dir: The path of the dataset in JSONL format.
- model_name_or_path & config_name & tokenizer_name: The path of the model.
- output_dir & output_model_dir: The path where the model is saved after training.
- train_filename & valid_filename & test_filename: The path where the dataset is stored.
- You can add "--expert_knowledge" at the end of the command to decide whether to add expert knowledge.
- You can add "--soft_prompt" at the end of the command to decide whether to use soft prompt.

```
python codebert_prompt.py
  --label_category 2
  --num_train_epochs 20 
  --train_batch_size 16 
  --valid_batch_size 16 
  --test_batch_size 16 
  --patience 10 
  --data_dir ../datasets 
  --model_name_or_path ../models/codebert-base 
  --config_name ../models/codebert-base 
  --tokenizer_name ../models/codebert-base 
  --output_dir ../saved-data/codebert-prompt 
  --output_model_dir ../saved-data/codebert-prompt 
  --train_filename ../datasets 
  --valid_filename ../datasets
  --test_filename ../datasets
  --do_train
  --do_test
```

Run the following command to start batch training.  The model is also saved at saved-data.

```
cd CodeBERT-PromptTuning
nohup bash codebert_prompt_batch_detection.sh > codebert_prompt_batch_detection.log
```





