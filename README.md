# bert-event-extraction
Pytorch Solution of Event Extraction Task using BERT on ACE 2005 corpus

## Prerequisites

1. Prepare **ACE 2005 dataset**. 

2. Use [nlpcl-lab/ace2005-preprocessing](https://github.com/nlpcl-lab/ace2005-preprocessing) to preprocess ACE 2005 dataset in the same format as the [data/sample.json](https://github.com/nlpcl-lab/bert-event-extraction/blob/master/data/sample.json). Then place it in the data directory as follows:
    ```
    ├── data
    │     └── test.json
    │     └── dev.json
    │     └── train.json
    │...
    ```

3. Install the packages.
   ```
   pip install pytorch==1.0 pytorch_pretrained_bert==0.6.1 numpy
   ```

## Usage

### Train
```
python train.py
```

### Evaluation
```
python eval.py --model_path=best_model.pt
```

### input
```
python input.py --model_path=best_model.pt
```
use it to input your sentence and return triggers and arguments
