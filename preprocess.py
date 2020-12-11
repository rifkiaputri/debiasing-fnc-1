import csv
import jsonlines


def csv_to_jsonl(file_in_body, file_in_stances, file_out):
    body_to_article = {}
    with open(file_in_body, 'r', encoding='utf-8', errors='ignore') as files:
        reader = csv.reader(files)
        next(reader, None)
        for item in reader:
            body_id, article = item
            body_to_article[body_id] = article

    combined_data = []
    with open(file_in_stances, 'r', encoding='utf-8', errors='ignore') as files:
        reader = csv.reader(files)
        next(reader, None)
        i = 0
        for item in reader:
            headline, body_id, stance, _ = item
            # if stance != 'unrelated':
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
    csv_to_jsonl('dataset/combine_test_bodies.csv', 'dataset/combine_test_stances_v2.csv', 'dataset/fnc.test.generated.jsonl')