from pathlib import Path
from pprint import pformat
from typing import Callable, Dict

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from sacred import Experiment

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.nn import DataParallel
from torch.utils.data import Dataset

from datasets import load_dataset, load_from_disk
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, TrainerCallback, Trainer, EvalPrediction,
)

ex = Experiment('instance-embedding-text')


@ex.config
def config():
    data_dir = 'data/text'
    model_dir = 'model/transformers'

    dataset_name = 'glue'
    task_name = 'cola'
    model_name = 'bert-base-cased'

    folder = task_name or dataset_name
    data_cache_dir = Path(data_dir) / folder
    model_cache_dir = Path(model_dir) / model_name
    output_dir = 'output'

    keys = ['sentence']
    metrics = ['matthew']
    with_embedding = True
    num_epochs = 3


@ex.command
def download_dataset(data_dir, dataset_name, task_name):
    datasets = load_dataset(dataset_name, task_name, cache_dir=Path(data_dir) / '.cache')
    datasets.save_to_disk(Path(data_dir) / (task_name or dataset_name))


class DictIndexDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, idx):
        item = self.dataset[idx]
        item['idx'] = idx  # idx within the split
        return item

    def __len__(self):
        return len(self.dataset)


class EmbeddingCallback(TrainerCallback):
    def on_step_begin(self, *_, model, **kwargs):
        # print('step begin')
        model.embed_optimizer.zero_grad()

    def on_step_end(self, *_, model, **kwargs):
        # print('step end')
        model.embed_optimizer.step()

    def on_epoch_end(self, *_, model, **kwargs):
        # print('epoch end')
        quantiles = [0, 0.25, 0.5, 0.75, 1]
        confidence = model.confidence
        print('quantile: ' + ' '.join(f'{q:.2%}' for q in np.quantile(confidence, quantiles)))


class EmbeddingTrainer(Trainer):
    def __init__(self,
                 model,
                 args: TrainingArguments,
                 train_dataset: Dataset,
                 embed_lr=1e10,
                 **kwargs):
        # idx -> idx within the split
        train_dataset = DictIndexDataset(train_dataset)

        device = args.device
        dataset_size = len(train_dataset)

        # inject embedding & embedding optimizer
        embed = nn.Embedding(dataset_size, 1, sparse=True)
        embed.weight.data.fill_(0.)
        embed = nn.Sequential(embed, nn.BatchNorm1d(1), nn.Sigmoid()).to(device)
        model.embed = embed
        model.embed_optimizer = optim.SGD(embed.parameters(), lr=embed_lr)

        # confidence list for evaluation
        model.indices = torch.arange(dataset_size).long().to(device)
        setattr(model.__class__, 'confidence',
                property(lambda model: model.embed(model.indices).detach().cpu().numpy().squeeze()))

        super(EmbeddingTrainer, self).__init__(model=model, args=args, train_dataset=train_dataset, **kwargs)
        self.add_callback(EmbeddingCallback())

    @staticmethod
    def compute_loss(model, inputs):
        labels = inputs.pop('labels')
        idx = inputs.pop('idx')
        # embedding
        embed = model.module.embed if isinstance(model, DataParallel) else model.embed
        confidence = embed(idx)
        # power_transformation
        outputs = model(**inputs)
        logits = confidence * outputs.logits
        return F.cross_entropy(logits, labels)


@ex.capture
def get_datasets(data_cache_dir, model_cache_dir, model_name, keys):
    # data
    datasets = load_from_disk(data_cache_dir)

    # tokenization
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=model_cache_dir,
        use_fast=True,
    )
    datasets = datasets.map(
        function=lambda example: tokenizer(*(example[key] for key in keys),
                                           padding='max_length', max_length=128, truncation=True),
        batched=True, load_from_cache_file=True,
    )
    return datasets


@ex.capture
def get_vl_splits(dataset_name, task_name):
    if dataset_name == 'glue':
        if task_name == 'mnli':
            vl_splits = ['validation_matched', 'validation_mismatched']
        else:
            vl_splits = ['validation']
    else:
        vl_splits = ['test']
    return vl_splits


@ex.capture
def get_model(num_classes,
              model_cache_dir, model_name):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        cache_dir=model_cache_dir,
        num_labels=num_classes,
    )
    return model


@ex.capture
def get_training_args(output_dir, num_epochs, _run):
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        weight_decay=0.01,
        fp16=True,
    )
    return training_args


@ex.capture
def compute_metrics(metrics: [str]) -> Callable[[EvalPrediction], Dict[str, float]]:
    m = {
        'accuracy': accuracy_score,
        'f1': f1_score,
        'matthew': matthews_corrcoef,
    }
    return lambda p: {metric: m[metric](p.label_ids, p.predictions.argmax(1))
                      for metric in metrics}


@ex.capture
def run(model, training_args, dataset_tr, datasets_vl,
        with_embedding,  # configs
        _run, _log):
    trainer_cls = EmbeddingTrainer if with_embedding else Trainer
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=dataset_tr,
        compute_metrics=compute_metrics(),
    )

    print('training...')
    trainer.train()

    print('testing...')
    for dataset_vl in datasets_vl:
        results = trainer.evaluate(eval_dataset=dataset_vl)
        _run.log_scalar('results', results)
        _log.info(f'\n{pformat(results)}')

    if with_embedding:
        # final confidence
        confidence = model.confidence
        _run.log_scalar('confidence', confidence.tolist())
        # final prediction
        p = trainer.prediction_loop(trainer.get_eval_dataloader(dataset_tr), '')
        prediction = p.predictions.argmax(1)
        _run.log_scalar('prediction', prediction.tolist())


@ex.main
def main():
    datasets = get_datasets()
    classes = datasets['train'].features['label'].names
    num_classes = len(classes)
    vl_splits = get_vl_splits()

    # split
    dataset_tr = datasets['train']
    datasets_vl = [datasets[vl] for vl in vl_splits]

    # model
    model = get_model(num_classes)
    training_args = get_training_args()

    run(model, training_args, dataset_tr, datasets_vl)


if __name__ == '__main__':
    ex.run_commandline()
