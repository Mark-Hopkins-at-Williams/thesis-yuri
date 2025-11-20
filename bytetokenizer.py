# bare minimum functionality ....

class Tokens:
    def __init__(self, input_ids):
        self.input_ids = input_ids 
    
    def __len__(self): 
        return len(self.input_ids)

class ByteTokenizer:
    def __init__(self, encoding):
        self.encoding = encoding

    def __call__(self, sentence):
        byted = sentence.encode(self.encoding)
        inputs = Tokens(list(byted))
        return inputs
