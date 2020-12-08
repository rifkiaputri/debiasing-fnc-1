import csv
import jsonlines


def csv_to_jsonl(file_in_body, file_in_stances, file_out):
    body_to_article = {}
    with open(file_in_body, 'r', encoding='utf-8') as files:
        reader = csv.reader(files)
        next(reader, None)
        for item in reader:
            body_id, article = item
            body_to_article[body_id] = article

    combined_data = []
    with open(file_in_stances, 'r', encoding='utf-8') as files:
        reader = csv.reader(files)
        next(reader, None)
        i = 0
        for item in reader:
            headline, body_id, stance = item
            if stance != 'unrelated':
                combined_data.append({
                    'gold_label': stance,
                    'evidence': body_to_article[body_id],
                    'claim': headline,
                    'id': i,
                })
                i += 1

    with jsonlines.open(file_out, mode='w') as writer:
        for item in combined_data:
            writer.write(item)


if __name__ == '__main__':
    csv_to_jsonl('dataset/train_bodies.csv', 'dataset/train_stances.csv', 'dataset/fnc.train.no-unrel.jsonl')
    csv_to_jsonl('dataset/competition_test_bodies.csv', 'dataset/competition_test_stances.csv', 'dataset/fnc.test.no-unrel.jsonl')