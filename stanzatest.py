import json
import stanza
from pathlib import Path

# stanza.download("tr")

nlp = stanza.Pipeline(
    lang="tr",
    processors="tokenize,pos,lemma,depparse",
    use_gpu=False
)

input_path = Path("dataset/tr-dev-v1.1.json")
output_path = Path("dataset_with_sentence_graphs.json")

with input_path.open("r", encoding="utf-8") as f:
    data = json.load(f)


def safe_lemma(word):
    if word.lemma is None or str(word.lemma).strip() == "":
        return word.text.lower()
    return word.lemma


def parse_text_to_sentence_graphs(text: str):
    doc = nlp(text)
    parsed_sentences = []

    for sent_idx, sent in enumerate(doc.sentences, start=1):
        tokens = []
        dependencies = []
        root_token_id = None
        root_token_text = None

        for w in sent.words:
            token_info = {
                "id": w.id,
                "text": w.text,
                "lemma": safe_lemma(w),
                "upos": w.upos,
                "xpos": w.xpos,
                "feats": w.feats,
                "head": w.head,
                "deprel": w.deprel
            }
            tokens.append(token_info)

            if w.deprel == "root":
                root_token_id = w.id
                root_token_text = w.text

        for head, rel, dep in sent.dependencies:
            dependencies.append({
                "head_id": 0 if head.id is None else head.id,
                "head_text": "ROOT" if head.id is None else head.text,
                "deprel": rel,
                "dep_id": dep.id,
                "dep_text": dep.text
            })

        parsed_sentences.append({
            "sentence_id": sent_idx,
            "text": sent.text,
            "root_token_id": root_token_id,
            "root_token_text": root_token_text,
            "tokens": tokens,
            "dependencies": dependencies
        })

    return parsed_sentences


for article in data["data"]:
    for paragraph in article["paragraphs"]:
        paragraph["context_sentences_parsed"] = parse_text_to_sentence_graphs(
            paragraph["context"]
        )

        for qa in paragraph["qas"]:
            qa["question_sentences_parsed"] = parse_text_to_sentence_graphs(
                qa["question"]
            )

with output_path.open("w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"Saved to: {output_path}")