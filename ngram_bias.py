import csv
import sys
import torch
import torch.optim as optim
import jsonlines
from tqdm import tqdm


def get_word_loss(prob, thres, prob_float, thres_float):
    if prob_float > thres_float:
        return torch.mean((prob - thres)**2)
    return torch.mean((0)**2)


word_files = [
    ('dataset/top_50_unigram_agree.csv', 'agree'),
    ('dataset/top_50_unigram_disagree.csv', 'disagree'),
    ('dataset/top_50_unigram_discuss.csv', 'discuss')
]
freq_data = {'agree': [], 'disagree': [], 'discuss': []}
for file, label in word_files:
    with open(file, 'r', encoding='utf-8') as files:
        reader = csv.reader(files)
        for item in reader:
            bigram, lmi, p_lw, freq, _, _, _ = item
            freq_data[label].append((bigram, p_lw))


fp = open('dataset/fnc.train.no-unrel.jsonl', "r", encoding='utf-8')
reader = jsonlines.Reader(fp)

train_claims = []
train_sentence_label = []
probs_tensor = []
probs_float = []
for dictionary in tqdm(reader.iter()):
    label, claim = dictionary['gold_label'], dictionary['claim']
    train_sentence_label.append(label)
    train_claims.append(claim)
    probs_data = [prob for tok, prob in freq_data[label] if tok in claim]
    max_prob = 0
    for prob in probs_data:
        if prob > max_prob:
            max_prob = prob
    probs_tensor.append(torch.Tensor([max_prob]))
    probs_float.append(max_prob)

x = torch.nn.Parameter(torch.zeros(len(train_sentence_label)))
if len(sys.argv) < 2:
    lr = 1
else:
    lr = float(sys.argv[1])
optimizer = optim.Adam(list([x]), lr=lr)
epochs = 2000
thres = torch.tensor([0.2])
thres_float = 0.2
for i in tqdm(range(epochs)):
    optimizer.zero_grad()
    loss = 0.0
    losses = []
    for prob, prob_float in zip(probs_tensor, probs_float):
        word_loss = get_word_loss(prob, thres, prob_float, thres_float)
        losses.append(word_loss)
    loss = sum(losses) + (1e-10 * torch.norm(x))
    loss.backward()
    optimizer.step()
    for p in x:
        p.data.clamp_(0)
    print(loss.data)
    print(sum(x.data.numpy().tolist()))


with open('parameters_' + str(epochs) + '.txt', 'w') as f:
    for claim, weight, label in zip(train_claims, x.data.numpy().tolist(), train_sentence_label):
        f.write(claim + "\n")
        f.write(str(weight) + "\n")
    f.close()


print('Write csv data...')
with open('dataset/fnc.train.no-unrel.weight.csv', 'w', encoding='utf-8', newline='\n') as out_csv:
    writer = csv.writer(out_csv)
    for claim, weight, label in zip(train_claims, x.data.numpy().tolist(), train_sentence_label):
        writer.writerow([claim, weight, label])
