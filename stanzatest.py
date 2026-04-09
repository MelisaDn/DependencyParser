import json
import stanza
from pathlib import Path

# stanza.download("tr")

nlp = stanza.Pipeline(
    lang="tr",
    processors="tokenize,pos,lemma,depparse",
    use_gpu=True
)

input_path = Path("dataset/morph-disamb-tr-train-v1.1.json")
output_path = Path("dataset/dependencies-tr-train-v1.1.json")


def parse_sentence_dependencies(text: str) -> dict:
    """
    Parses one sentence and returns only dependency information.
    """
    result = {
        "sentence": text,
        "dependencies": []
    }

    if text is None or not text.strip():
        return result

    doc = nlp(text)

    if not doc.sentences:
        return result

    sent = doc.sentences[0]

    for word in sent.words:
        result["dependencies"].append({
            "dep_id": word.id,
            "head_id": word.head,
            "deprel": word.deprel
        })

    return result


with input_path.open("r", encoding="utf-8") as f:
    data = json.load(f)


for article in data.get("data", []):
    for paragraph in article.get("paragraphs", []):
        context_sentences = paragraph.get("context_sentences", [])

        paragraph["context_dependencies"] = [
            parse_sentence_dependencies(sentence_text)
            for sentence_text in context_sentences
        ]

        for qa in paragraph.get("qas", []):
            question_text = qa.get("question", "")

            qa["question_dependencies"] = parse_sentence_dependencies(question_text)


with output_path.open("w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"Saved to: {output_path}")