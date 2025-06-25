import unittest
from corpora import Bitext, MixtureOfBitexts, TokenizedMixtureOfBitexts
from finetune import tokenize, prepare_tokenizer_and_model
from transformers import AutoTokenizer
from torch import tensor


class TestUtil(unittest.TestCase):
    def test_streaming_bitext(self):
        bitext = Bitext("test_files/lang1.txt", "test_files/lang2.txt")
        expected = [
            ("The cat chased the mouse.", "Le chat a poursuivi la souris."),
            ("She reads a book.", "Elle lit un livre."),
            ("They play soccer.", "Ils jouent au football."),
            ("I ate dinner.", "J’ai dîné."),
            ("He drinks coffee.", "Il boit du café."),
            ("We watched a movie.", "Nous avons regardé un film."),
            ("The dog barked at strangers.", "Le chien a aboyé sur des inconnus."),
            ("You wrote a letter.", "Tu as écrit une lettre."),
            ("John opened the door.", "John a ouvert la porte."),
            ("The teacher gave homework.", "Le professeur a donné des devoirs."),
            ("Sarah paints pictures.", "Sarah peint des tableaux."),
            ("The baby kicked the ball.", "Le bébé a frappé le ballon."),
            ("Tom fixed the bike.", "Tom a réparé le vélo."),
            ("Emma baked a cake.", "Emma a fait un gâteau."),
            ("The child drew a star.", "L’enfant a dessiné une étoile."),
            ("My brother broke the window.", "Mon frère a cassé la fenêtre."),
            ("Lisa hugged her friend.", "Lisa a serré son amie dans ses bras."),
            ("Mark answered the question.", "Mark a répondu à la question."),
            ("The chef cooked a meal.", "Le chef a cuisiné un repas."),
            ("They built a house.", "Ils ont construit une maison."),
        ]
        result = [line for line in bitext]
        self.assertEqual(expected, result)

    def test_mixture_of_bitexts(self):
        bitext1 = Bitext("test_files/lang1.txt", "test_files/lang2.txt")
        bitext2 = Bitext("test_files/lang1.txt", "test_files/lang3.txt")
        mix = MixtureOfBitexts(
            {("lang1", "lang2"): bitext1, ("lang1", "lang3"): bitext2}, 3
        )
        batch = mix.next_batch()
        expected1 = (
            ("The cat chased the mouse.", "She reads a book.", "They play soccer."),
            (
                "Le chat a poursuivi la souris.",
                "Elle lit un livre.",
                "Ils jouent au football.",
            ),
            "lang1",
            "lang2",
        )
        expected2 = (
            ("The cat chased the mouse.", "She reads a book.", "They play soccer."),
            (
                "Die Katze jagte die Maus.",
                "Sie liest ein Buch.",
                "Sie spielen Fußball.",
            ),
            "lang1",
            "lang3",
        )
        self.assertIn(batch, [expected1, expected2])

    def test_mixture_of_bitexts2(self):
        text_files = {
            "lang1": "test_files/lang1.txt",
            "lang2": "test_files/lang2.txt",
            "lang3": "test_files/lang3.txt",
        }
        mix = MixtureOfBitexts.create_from_files(
            text_files, [("lang1", "lang2"), ("lang1", "lang3")], 3
        )
        batch = mix.next_batch()
        expected1 = (
            ("The cat chased the mouse.", "She reads a book.", "They play soccer."),
            (
                "Le chat a poursuivi la souris.",
                "Elle lit un livre.",
                "Ils jouent au football.",
            ),
            "lang1",
            "lang2",
        )
        expected2 = (
            ("The cat chased the mouse.", "She reads a book.", "They play soccer."),
            (
                "Die Katze jagte die Maus.",
                "Sie liest ein Buch.",
                "Sie spielen Fußball.",
            ),
            "lang1",
            "lang3",
        )
        self.assertIn(batch, [expected1, expected2])

    def test_tokenize(self):
        text_files = {"lang1": "test_files/lang1.txt", "lang2": "test_files/lang2.txt"}
        mix = MixtureOfBitexts.create_from_files(text_files, [("lang1", "lang2")], 3)
        base_model = "facebook/nllb-200-distilled-600M"
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        lang1_sents, lang2_sents, lang1, lang2 = mix.next_batch()
        tokenized = tokenize(lang1_sents, lang1, tokenizer, 100)
        expected_ids = tensor(
            [
                [3, 1617, 7875, 228, 55501, 349, 227879, 248075, 2],
                [3, 11873, 272, 22665, 9, 28487, 248075, 2, 1],
                [3, 13710, 18379, 43583, 2299, 248075, 2, 1, 1],
            ]
        )
        expected_mask = tensor(
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 1, 1, 0, 0],
            ]
        )
        self.assertEqual(tokenized["input_ids"].tolist(), expected_ids.tolist())
        self.assertEqual(tokenized["attention_mask"].tolist(), expected_mask.tolist())
        tokens = [
            tokenizer.convert_ids_to_tokens(seq) for seq in tokenized["input_ids"]
        ]
        expected_tokens = [
            ["<unk>", "▁The", "▁cat", "▁ch", "ased", "▁the", "▁mouse", ".", "</s>"],
            ["<unk>", "▁She", "▁re", "ads", "▁a", "▁book", ".", "</s>", "<pad>"],
            ["<unk>", "▁They", "▁play", "▁soc", "cer", ".", "</s>", "<pad>", "<pad>"],
        ]
        self.assertEqual(tokens, expected_tokens)
        
    def test_tokenize_alt_pad(self):
        text_files = {"eng_Latn": "test_files/lang1.txt", "fra_Latn": "test_files/lang2.txt"}
        mix = MixtureOfBitexts.create_from_files(text_files, [("eng_Latn", "fra_Latn")], 3)
        base_model = "facebook/nllb-200-distilled-600M"
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        lang1_sents, lang2_sents, lang1, lang2 = mix.next_batch()
        tokenized = tokenize(lang1_sents, lang1, tokenizer, 100,
                             alt_pad_token=-100)
        expected_ids = tensor(
            [
                [256047, 1617, 7875, 228, 55501, 349, 227879, 248075, 2],
                [256047, 11873, 272, 22665, 9, 28487, 248075, 2, -100],
                [256047, 13710, 18379, 43583, 2299, 248075, 2, -100, -100],
            ]
        )
        expected_mask = tensor(
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 1, 1, 0, 0],
            ]
        )
        self.assertEqual(tokenized["input_ids"].tolist(), expected_ids.tolist())
        self.assertEqual(tokenized["attention_mask"].tolist(), expected_mask.tolist())
        
    def test_tokenized_mixture_of_bitexts(self):
        text_files = {"eng_Latn": "test_files/lang1.txt", "fra_Latn": "test_files/lang2.txt"}
        mix = MixtureOfBitexts.create_from_files(text_files, [("eng_Latn", "fra_Latn")], 3)
        base_model = "facebook/nllb-200-distilled-600M"
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        tmob = TokenizedMixtureOfBitexts(mix, tokenizer, max_length=128)
        lang1_token_ids, lang2_token_ids, lang1_mask, lang2_mask = tmob.next_batch()
        expected_lang1_token_ids = tensor([[256047,   1617,   7875,    228,  55501,    349, 227879, 248075,      2],
                                           [256047,  11873,    272,  22665,      9,  28487, 248075,      2,      1],
                                           [256047,  13710,  18379,  43583,   2299, 248075,      2,      1,      1]])
        expected_lang2_token_ids = tensor([[256057,   1181,  32779,      9, 170684,    356,     82,    324,  40284,  248075,      2],
                                           [256057,  19945,   6622,    159,  68078, 248075,      2,   -100,   -100,    -100,   -100],
                                           [256057,  21422,   5665,    138,   1166,  96236, 248075,      2,   -100,    -100,   -100]])
        expected_lang1_mask = tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1],
                                      [1, 1, 1, 1, 1, 1, 1, 1, 0],
                                      [1, 1, 1, 1, 1, 1, 1, 0, 0]])
        expected_lang2_mask = tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                      [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                                      [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]])
        self.assertEqual(lang1_token_ids.tolist(), expected_lang1_token_ids.tolist())
        self.assertEqual(lang2_token_ids.tolist(), expected_lang2_token_ids.tolist())
        self.assertEqual(lang1_mask.tolist(), expected_lang1_mask.tolist())
        self.assertEqual(lang2_mask.tolist(), expected_lang2_mask.tolist())
        
    def test_tokenized_mixture_of_bitexts_truncated(self):
        text_files = {"eng_Latn": "test_files/lang1.txt", "fra_Latn": "test_files/lang2.txt"}
        mix = MixtureOfBitexts.create_from_files(text_files, [("eng_Latn", "fra_Latn")], 3)
        base_model = "facebook/nllb-200-distilled-600M"
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        tmob = TokenizedMixtureOfBitexts(mix, tokenizer, max_length=8)
        lang1_token_ids, lang2_token_ids, lang1_mask, lang2_mask = tmob.next_batch()
        expected_lang1_token_ids = tensor([[256047,   1617,   7875,    228,  55501,    349, 227879, 2],
                                           [256047,  11873,    272,  22665,      9,  28487, 248075,      2],
                                           [256047,  13710,  18379,  43583,   2299, 248075,      2,      1]])
        expected_lang2_token_ids = tensor([[256057,   1181,  32779,      9, 170684,    356,     82,    2],
                                           [256057,  19945,   6622,    159,  68078, 248075,      2,   -100],
                                           [256057,  21422,   5665,    138,   1166,  96236, 248075,      2]])
        expected_lang1_mask = tensor([[1, 1, 1, 1, 1, 1, 1, 1],
                                      [1, 1, 1, 1, 1, 1, 1, 1],
                                      [1, 1, 1, 1, 1, 1, 1, 0]])
        expected_lang2_mask = tensor([[1, 1, 1, 1, 1, 1, 1, 1],
                                      [1, 1, 1, 1, 1, 1, 1, 0],
                                      [1, 1, 1, 1, 1, 1, 1, 1]])
        self.assertEqual(lang1_token_ids.tolist(), expected_lang1_token_ids.tolist())
        self.assertEqual(lang2_token_ids.tolist(), expected_lang2_token_ids.tolist())
        self.assertEqual(lang1_mask.tolist(), expected_lang1_mask.tolist())
        self.assertEqual(lang2_mask.tolist(), expected_lang2_mask.tolist())
        

if __name__ == "__main__":
    unittest.main()
