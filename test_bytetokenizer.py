import unittest
from bytetokenizer import ByteTokenizer


class TestTokenization(unittest.TestCase):
    
    def test_byte_tokenizer1(self):
        tokenizer = ByteTokenizer('utf-8')
        result = tokenizer('the cat')
        expected = {'input_ids': [116, 104, 101, 32, 99, 97, 116]}
        self.assertEqual(result, expected)

    def test_byte_tokenizer2(self):
        tokenizer = ByteTokenizer('utf-8')
        result = tokenizer('the 🐱')
        expected = {'input_ids': [116, 104, 101, 32, 240, 159, 144, 177]}
        self.assertEqual(result, expected)
 

if __name__ == "__main__":
    unittest.main()
