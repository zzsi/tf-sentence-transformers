from tf_sentence_transformers import SentenceTransformer


def test_call():
    layer = SentenceTransformer.from_pretrained(
        "sentence-transformers/all-MiniLM-L6-v2", from_pt=False
    )
    inputs = [
        ["This is a test sentence."],
        ["This is another test sentence."],
    ]
    embedding = layer(inputs)
    assert embedding.shape == (2, 384)
