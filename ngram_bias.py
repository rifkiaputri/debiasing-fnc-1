import csv
import torch
import torch.optim as optim
import torch.nn as nn
import jsonlines
from tqdm import tqdm


word_files = [
    ('dataset/top_1000_unigram_agree.csv', 'agree'),
    ('dataset/top_1000_unigram_disagree.csv', 'disagree'),
    ('dataset/top_1000_unigram_discuss.csv', 'discuss')
]
freq_data = {'agree': [], 'disagree': [], 'discuss': []}
for file, label in word_files:
    with open(file, 'r', encoding='utf-8') as files:
        reader = csv.reader(files)
        for item in reader:
            bigram, lmi, p_lw, freq, _, _, _, score_a, score_b = item
            freq_data[label].append((bigram, p_lw, score_a, score_b))


fp = open('dataset/fnc.train.no-unrel.jsonl', "r", encoding='utf-8')
reader = jsonlines.Reader(fp)

train_claims = []
train_sentence_label = []
id_evidence = []
prob_tensor = []
score_a_tensor = []
score_b_tensor = []
zeros_instances = []
for dictionary in tqdm(reader.iter()):
    label, claim, id, evidence = dictionary['gold_label'], dictionary['claim'], dictionary['id'], dictionary['evidence']
    probs_data = [(float(prob), int(score_a), int(score_b)) for tok, prob, score_a, score_b in freq_data[label] if tok in claim]
    if len(probs_data) > 0:
        max_prob, max_a, max_b = 0, 0, 1
        for prob, score_a, score_b in probs_data:
            if prob > max_prob:
                max_prob = prob
                max_a = score_a
                max_b = score_b
        prob_tensor.append(torch.tensor([max_prob]))
        score_a_tensor.append(torch.tensor([max_a]))
        score_b_tensor.append(torch.tensor([max_b]))
        train_sentence_label.append(label)
        train_claims.append(claim)
        id_evidence.append((id, evidence))
    else:
        zeros_instances.append(dictionary)

x = torch.nn.Parameter(torch.zeros(len(train_sentence_label)), requires_grad=True)
lr = 1
optimizer = optim.Adam([x], lr=lr)
epochs = 2000
thres = torch.tensor([0.2])
loss_fct = nn.MSELoss()
pbar = tqdm(range(epochs))
for i in pbar:
    optimizer.zero_grad()
    losses = []
    for j in range(len(score_a_tensor)):
        pred = (score_a_tensor[j] * (1 + x[j])) / (score_b_tensor[j] * (1 + x[j]))
        losses.append(loss_fct(pred, thres))
    loss = sum(losses) + (1e-10 * torch.norm(x))
    loss.backward()
    optimizer.step()
    for p in x:
        p.data.clamp_(0)
    pbar.set_description("Weight %s" % str(sum(x.data.numpy().tolist())))


with torch.no_grad():
    for j in range(len(score_a_tensor)):
        pred = (score_a_tensor[j] * (1 + x[j])) / (score_b_tensor[j] * (1 + x[j]))
        print('Pred:', pred)
        print('Prob:', prob_tensor[j])
        print()


print('Write output data...')
with jsonlines.open('dataset/fnc.train.no-unrel.weight_zeros.jsonl', mode='w') as writer:
    for claim, weight, label, id_ev in zip(train_claims, x.data.numpy().tolist(), train_sentence_label, id_evidence):
        writer.write({
            'gold_label': label,
            'evidence': id_ev[1],
            'claim': claim,
            'id': id_ev[0],
            'weight': weight
        })

    for item in zeros_instances:
        item['weight'] = 0.0
        writer.write(item)
