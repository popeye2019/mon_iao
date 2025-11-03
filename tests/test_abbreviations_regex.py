import unittest
import sys
from pathlib import Path

# Ensure project root is on sys.path when running the test directly
ROOT = Path(__file__).resolve().parents[1]
# Prepend to prioritize local sources
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.text_normalize import expand_abbreviations


class TestAbbreviationsRegex(unittest.TestCase):
    def test_saas(self):
        txt = "Solution en SaaS pour la supervision"
        out = expand_abbreviations(txt)
        self.assertIn("Software as a Service (SaaS)", out)

    def test_st_ste(self):
        txt = "St Michel et Ste Marie"
        out = expand_abbreviations(txt)
        self.assertIn("Saint Michel", out)
        self.assertIn("Sainte Marie", out)

    def test_pH(self):
        txt = "Le pH doit être 7.2"
        out = expand_abbreviations(txt)
        # Le dictionnaire mappe PH -> "Potentiel hydrogène (pH)"
        self.assertIn("Potentiel hydrogène (pH)", out)

    def test_R_bullet(self):
        txt = "- R: Pompe 2 en défaut"
        out = expand_abbreviations(txt)
        self.assertIn("Poste de relevage (R):", out)


if __name__ == "__main__":
    unittest.main()
