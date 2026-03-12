import json
import stanza

nlp = stanza.Pipeline(
    lang="tr",
    processors="tokenize,pos,lemma,depparse",
    use_gpu=False
)

with open("dataset/tr-dev-v1.1.json", "r", encoding="utf-8") as f:
    data = json.load(f)

sample_context = data["data"][0]["paragraphs"][0]["context"]
sample_question = data["data"][0]["paragraphs"][0]["qas"][0]["question"]

print("CONTEXT:")
doc = nlp(sample_context)
for sent_idx, sent in enumerate(doc.sentences, start=1):
    print(f"\n--- Context Sentence {sent_idx} ---")
    print("Sentence text:", sent.text)
    for w in sent.words:
        print(
            f"id={w.id}\ttext={w.text}\thead={w.head}\tdeprel={w.deprel}"
            f"\tlemma={w.lemma}\tupos={w.upos}\tfeats={w.feats}"
        )

print("\nQUESTION:")
doc_q = nlp(sample_question)
for sent_idx, sent in enumerate(doc_q.sentences, start=1):
    print(f"\n--- Question Sentence {sent_idx} ---")
    print("Sentence text:", sent.text)
    for w in sent.words:
        print(
            f"id={w.id}\ttext={w.text}\thead={w.head}\tdeprel={w.deprel}"
            f"\tlemma={w.lemma}\tupos={w.upos}\tfeats={w.feats}"
        )