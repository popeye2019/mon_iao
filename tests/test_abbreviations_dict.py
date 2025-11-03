import unittest
import sys
from pathlib import Path

# Ensure project root is on sys.path when running the test directly
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.text_normalize import expand_abbreviations


class TestAbbreviationsDict(unittest.TestCase):
    def test_simple_abbr(self):
        txt = "Le RES est plein, la STEP fonctionne."
        out = expand_abbreviations(txt)
        self.assertIn("Réservoir (RES)", out)
        self.assertIn("Station d'épuration (STEP)", out)

    def test_case_insensitive(self):
        txt = "Le res et la step et le PR"
        out = expand_abbreviations(txt)
        # Résultats attendus en conservant la casse rencontrée entre parenthèses
        self.assertIn("Réservoir (res)", out)
        self.assertIn("Station d'épuration (step)", out)
        self.assertIn("Poste de relevage (PR)",out)


if __name__ == "__main__":
    unittest.main()
