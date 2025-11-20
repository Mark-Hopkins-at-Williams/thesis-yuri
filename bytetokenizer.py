class ByteTokenizer:
    def __init__(self, encoding='utf-8'):
        self.encoding = encoding

    def __call__(self, sentence):
        byted = sentence.encode(self.encoding)
        inputs = {'input_ids': list(byted)}
        return inputs
