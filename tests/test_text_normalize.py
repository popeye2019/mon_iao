import unittest
import sys
from pathlib import Path

# Ensure project root on sys.path when running directly
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.text_normalize import expand_abbreviations


E = chr(233)

class TestExpandAbbreviations(unittest.TestCase):
    def test_dict_expansion(self):
        txt = "Le RES est plein, la STEP fonctionne."
        out = expand_abbreviations(txt)
        self.assertIn("R\u00e9servoir (RES)", out)
        self.assertIn("Station d'\u00e9puration (STEP)", out)

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



