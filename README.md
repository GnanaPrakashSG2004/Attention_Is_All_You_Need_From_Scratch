# File Structure:
```
.
├── Assignment_2.pdf
├── corpus
│   ├── dev.en
│   ├── dev.fr
│   ├── processed_dev.en
│   ├── processed_dev.fr
│   ├── processed_test.en
│   ├── processed_test.fr
│   ├── processed_train.en
│   ├── processed_train.fr
│   ├── test.en
│   ├── test.fr
│   ├── train.en
│   └── train.fr
├── metrics
│   └── testbleu.txt
├── plots.ipynb
├── README.md
├── Report.pdf
└── src
    ├── decoder.py
    ├── encoder.py
    ├── __init__.py
    ├── test.py
    ├── train.py
    └── utils.py
```
---

# Instructions to run the code:
- All scripts can be run by staying in the current (root) directory.

## Processing the corpus for the first time:
- Run the following command to process the corpus for the first time:
```bash
python -m src.utils
```
- This will create the processed files in the `corpus` directory with `processed_` prefix.
---

## Training the model:
- Run the following command to train the model:
```bash
python -m src.train
```
- This will train the model and save the model in the `weights` directory.
- To change the hyperparameters, you can pass them as arguments to the above command. For example:
```bash
python -m src.train --epochs 10 --batch_size 64
```
- To see all the hyperparameters, you can run:
```bash
python -m src.train --help
```
---

## Testing the model:
- Run the following command to test the model:
```bash
python -m src.test
```
- This will test the model and save the BLEU score for the test set in the `metrics` directory.
---

## Loading the model:
- The file `weights/transformer.pt` is a state dict containing various parameters of the model and other information computed during the training process.
- A list of these parameters is as follows:
  - `args`: Hyperparameters of the training run.
  - `model_state_dict`: The state dict of the model.
  - `optimizer_state_dict`: The state dict of the optimizer.
  - `en_seq_len`: The sequence length of the English train dataset.
  - `fr_seq_len`: The sequence length of the French train dataset.
  - `word2idx_en`: The word to index mapping of the English train dataset.
  - `word2idx_fr`: The word to index mapping of the French train dataset.
  - `idx2word_en`: The index to word mapping of the English train dataset.
  - `idx2word_fr`: The index to word mapping of the French train dataset.
  - `train_losses`: The training losses.
  - `dev_losses`: The validation losses.
  - `dev_bleu`: The BLEU score on the validation set.
  - `dev_rouge`: The ROUGE score on the validation set.
- To load the model, you can use the following code:
```python
model = Transformer(args) # Pass the necessary args
model.load_state_dict(torch.load('weights/transformer.pt')['model_state_dict'])
```
---

## Implementation Assumptions:
- Used a sequence length of `77` for faster training during hyperparameter tuning.
- If a sequence length of `77` is used, the model will assume that apart from the `<BOS>` and `<EOS>` tokens, the maximum length of the sentence is `77`.
- The model will pad the sentences to a length of `77` if the sentence is shorter than `77`.
- The model will truncate the sentences to a length of `77` if the sentence is longer than `77`.
- The model will not learn the padding token.
- The position embeddings are not learned.
- The model uses the `Adam` optimizer with a learning rate of `0.0001`.
- The model uses the `CrossEntropyLoss` loss function.
---

## Pretrained Model weights:
- The main model with `6` blocks, `8` heads, `512`-dim embeddings and `0.1` has been trained for `10` epochs on the maximum sequence length of the training data.
  - The model can be downloaded from [here](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/punnavajhala_prakash_research_iiit_ac_in/EUBg3jIIvA9LkT0KmVLDpaoBFbR_9tlJUtN3sYW4gWgz7A?e=Bdfhb5)
- Some more models saved during the hyperparameter tuning process can be downloaded from the following links:
  - `6` blocks, `8` heads, `512`-dim embeddings and `0.1` dropout rate trained for `10` epochs with a sequence length of `77`:
    - [Link](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/punnavajhala_prakash_research_iiit_ac_in/EUA6ciXSphRBt_yq7D2wJKMBWrUc1Ln4hA8KbR81T3_tIg?e=8NuR50)
  - `6` blocks, `8` heads, `640`-dim embeddings and `0.2` dropout rate trained for `10` epochs with a sequence length of `77`:
    - [Link](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/punnavajhala_prakash_research_iiit_ac_in/Efs2HEKqXgVKuE0sZFy3ecABlcFCJeJGw0dDrwIOsxcV9Q?e=wjoUCX)
  - `8` blocks, `10` heads, `640`-dim embeddings and `0.1` dropout rate trained for `10` epochs with a sequence length of `77`:
    - [Link](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/punnavajhala_prakash_research_iiit_ac_in/EXtnIwQx6FJHtBNOULkK3ZgBf_hsxCQZrTTeuaNZ2vGiIw?e=hqrvMc)