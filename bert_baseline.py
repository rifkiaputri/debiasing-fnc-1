import argparse
import json
import pandas as pd
import logging
import sklearn
from my_classification import ClassificationModel, ClassificationArgs


def my_f1_score(y_true, y_pred):
    return sklearn.metrics.f1_score(y_true, y_pred, average='weighted')


def load_fever_jsonl(input_path, is_train=True):
    """
    Read list of objects from a JSON lines file.
    """
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.rstrip('\n|\r')))
    print('Loaded {} records from {}'.format(len(data), input_path))

    db_data = []
    if is_train:
        db_cols = ['claim', 'gold_label', 'weight']
    else:
        db_cols = ['claim', 'gold_label']
    for d in data:
        db_data.append([])
        for col in db_cols:
            db_data[-1].append(d.get(col, float('nan')))

    return db_data, db_cols


def load_fever_jsonl_two_texts(input_path, is_train=True):
    """
    Read list of objects from a JSON lines file.
    """
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.rstrip('\n|\r')))
    print('Loaded {} records from {}'.format(len(data), input_path))

    db_data = []
    if is_train:
        db_cols = ['claim', 'evidence', 'gold_label', 'weight']
    else:
        db_cols = ['claim', 'evidence', 'gold_label']
    for d in data:
        db_data.append([])
        for col in db_cols:
            db_data[-1].append(d.get(col, float('nan')))

    return db_data, db_cols


parser = argparse.ArgumentParser()
parser.add_argument('-t', '--task', type=str, required=True, choices=['fever', 'fnc', 'fever-2', 'fnc-2'], help='task type (fever/fnc)')
parser.add_argument('-o', '--out', type=str, required=True, help='output directory name')
parser.add_argument('-p', '--proj', type=str, required=True, help='wandb project name')
parser.add_argument('-d', '--dnum', type=int, required=True, help='cuda device num')
parser.add_argument('-l', '--lr', type=float, required=True, help='learning rate')
args = parser.parse_args()
print('Args:', args)

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger('transformers')
transformers_logger.setLevel(logging.WARNING)

# Preparing label
if args.task in ['fnc', 'fnc-2']:
    class_labels = ['agree', 'disagree']
    # class_labels = ['agree', 'disagree', 'discuss']
    class_weights = [1.00, 4.38]
    # class_weights = [2.42, 10.61, 1.00]
elif args.task in ['fever', 'fever-2']:
    # class_labels = ['SUPPORTS', 'REFUTES', 'NOT ENOUGH INFO']
    class_labels = ['SUPPORTS', 'REFUTES']
    class_weights = None
else:
    class_labels = []
    class_weights = None

# Preparing train and eval data
if args.task == 'fnc':
    db_data, db_cols = load_fever_jsonl('dataset/fnc.train.no-unrel.weight_zeros.jsonl', is_train=True)
    train_df = pd.DataFrame(db_data, columns=db_cols)
    train_df.columns = ['text', 'labels', 'weight']
    train_df = train_df[train_df.labels != 'unrelated']
    train_df = train_df[train_df.labels != 'discuss']
    db_data, db_cols = load_fever_jsonl('dataset/fnc.test.no-unrel.jsonl', is_train=False)
    eval_df = pd.DataFrame(db_data, columns=db_cols)
    eval_df.columns = ['text', 'labels']
    eval_df = eval_df[eval_df.labels != 'unrelated']
    eval_df = eval_df[eval_df.labels != 'discuss']
elif args.task == 'fnc-2':
    db_data, db_cols = load_fever_jsonl_two_texts('dataset/fnc.train.no-unrel.weight_zeros.jsonl', is_train=True)
    train_df = pd.DataFrame(db_data, columns=db_cols)
    train_df.columns = ['text_a', 'text_b', 'labels', 'weight']
    train_df = train_df[train_df.labels != 'unrelated']
    train_df = train_df[train_df.labels != 'discuss']
    db_data, db_cols = load_fever_jsonl_two_texts('dataset/fnc.test.no-unrel.jsonl', is_train=False)
    eval_df = pd.DataFrame(db_data, columns=db_cols)
    eval_df.columns = ['text_a', 'text_b', 'labels']
    eval_df = eval_df[eval_df.labels != 'unrelated']
    eval_df = eval_df[eval_df.labels != 'discuss']
elif args.task == 'fever':
    db_data, db_cols = load_fever_jsonl('dataset/fever/fever.train.jsonl', is_train=False)
    train_df = pd.DataFrame(db_data, columns=db_cols)
    train_df.columns = ['text', 'labels']
    train_df = train_df[train_df.labels != 'NOT ENOUGH INFO']
    db_data, db_cols = load_fever_jsonl('dataset/fever/fever.dev.jsonl', is_train=False)
    eval_df = pd.DataFrame(db_data, columns=db_cols)
    eval_df.columns = ['text', 'labels']
    eval_df = eval_df[eval_df.labels != 'NOT ENOUGH INFO']
elif args.task == 'fever-2':
    db_data, db_cols = load_fever_jsonl_two_texts('dataset/fever/fever.train.jsonl', is_train=True)
    train_df = pd.DataFrame(db_data, columns=db_cols)
    train_df.columns = ['text_a', 'text_b', 'labels', 'weight']
    train_df = train_df[train_df.labels != 'NOT ENOUGH INFO']
    db_data, db_cols = load_fever_jsonl_two_texts('dataset/fever/fever.dev.jsonl', is_train=False)
    eval_df = pd.DataFrame(db_data, columns=db_cols)
    eval_df.columns = ['text_a', 'text_b', 'labels']
    eval_df = eval_df[eval_df.labels != 'NOT ENOUGH INFO']
else:
    raise NotImplementedError('Undefined task.')

# Optional model configuration
model_args = ClassificationArgs()
model_args.num_train_epochs = 10
model_args.train_batch_size = 64
model_args.eval_batch_size = 32
model_args.learning_rate = args.lr
model_args.output_dir = '/mnt/nas2/kiki/debiasing-fnc-1/outputs/' + args.out + '/'
model_args.best_model_dir = '/mnt/nas2/kiki/debiasing-fnc-1/outputs/' + args.out + '/best-model/'
model_args.labels_list = class_labels
model_args.wandb_project = args.proj
model_args.use_early_stopping = True
model_args.early_stopping_delta = 0.01
model_args.early_stopping_metric = 'f1'
model_args.early_stopping_metric_minimize = False
model_args.early_stopping_patience = 3
model_args.save_model_every_epoch = False
model_args.evaluate_during_training = True
model_args.best_eval_metric = 'f1'
model_args.manual_seed = 42
model_args.overwrite_output_dir = True

# Create a ClassificationModel
model = ClassificationModel(
    'bert',
    'bert-base-cased',
    num_labels=len(class_labels),
    weight=class_weights,
    args=model_args,
    cuda_device=args.dnum,
    wandb_run_name=args.out,
    has_bias_weight=True
)

# Train the model
model.train_model(train_df, eval_df=eval_df, acc=sklearn.metrics.accuracy_score, f1=my_f1_score)

# Predict best model
model = ClassificationModel(
    'bert',
    model_args.best_model_dir,
    num_labels=len(class_labels),
    weight=class_weights,
    args=model_args,
    cuda_device=args.dnum,
    wandb_run_name=args.out
)
result, model_outputs, wrong_predictions = model.eval_model(eval_df)
if 'text' in eval_df.columns:
    predictions, raw_outputs = model.predict(eval_df['text'].tolist())
else:
    texts_a = eval_df['text_a'].tolist()
    texts_b = eval_df['text_a'].tolist()
    predictions, raw_outputs = model.predict([[texts_a[i], texts_b[i]] for i in range(len(texts_a))])
eval_df['prediction'] = predictions
eval_df.to_csv(model_args.best_model_dir + 'bert-output.csv', index=False)
