import unittest

from app.text_normalize import expand_abbreviations


class TestExpandAbbreviations(unittest.TestCase):
    def test_dict_expansion(self):
        txt = "Le RES est plein, la STEP fonctionne."
        out = expand_abbreviations(txt)
        self.assertIn("Réservoir (RES)", out)
        self.assertIn("Station d'épuration (STEP)", out)

    def test_regex_st_ste(self):
        txt = "St Michel et Ste Marie"
        out = expand_abbreviations(txt)
        self.assertIn("Saint Michel", out)
        self.assertIn("Sainte Marie", out)

    def test_saas(self):
        txt = "Solution en SaaS pour la supervision"
        out = expand_abbreviations(txt)
        self.assertIn("Software as a Service (SaaS)", out)


if __name__ == "__main__":
    unittest.main()

