"""Run with:  python -m unittest discover -s tests -v"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import create_app  # noqa: E402
from src import features  # noqa: E402


class FeatureTests(unittest.TestCase):
    def test_extract(self):
        f = features.extract_features("BREAKING!!! This is NOT good. Terrible news!")
        self.assertEqual(f["exclamation_count"], 4)
        self.assertEqual(f["word_count"], 7)
        self.assertAlmostEqual(f["all_caps_pct"], round(2 / 7, 3))
        self.assertLess(f["sentiment_score"], 0)

    def test_negation_flips(self):
        self.assertGreater(features.sentiment_score("This is good"), features.sentiment_score("This is not good"))

    def test_empty(self):
        self.assertEqual(features.extract_features("")["word_count"], 0)


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = create_app().test_client()

    def test_health(self):
        self.assertEqual(self.client.get("/api/health").get_json()["status"], "ok")

    def test_index(self):
        self.assertEqual(self.client.get("/").status_code, 200)

    def test_short_text_uses_text_model(self):
        r = self.client.post("/api/predict", json={"text": "Says the state cut school funding every year."})
        j = r.get_json()
        self.assertEqual(r.status_code, 200)
        self.assertIn(j["verdict"].split()[0], {"LIKELY", "UNCERTAIN"})
        self.assertTrue(0 <= j["p_fake"] <= 1)
        self.assertIsNone(j["style_model"])
        self.assertIn("disclaimer", j)

    def test_long_text_uses_both(self):
        text = ("WOW!!! SHOCKING secret scandal!!! " * 60)
        j = self.client.post("/api/predict", json={"text": text}).get_json()
        self.assertEqual(sorted(j["models_used"]), ["style", "text"])

    def test_sliders(self):
        j = self.client.post("/api/predict", json={"features": {
            "word_count": 500, "sentiment_score": 0.9, "exclamation_count": 14, "all_caps_pct": 0.28}}).get_json()
        calm = self.client.post("/api/predict", json={"features": {
            "word_count": 1500, "sentiment_score": 0.0, "exclamation_count": 1, "all_caps_pct": 0.03}}).get_json()
        self.assertGreater(j["p_fake"], calm["p_fake"])

    def test_bad_input(self):
        self.assertEqual(self.client.post("/api/predict", json={}).status_code, 400)
        self.assertEqual(self.client.post("/api/predict", json={"text": "   "}).status_code, 400)
        self.assertEqual(self.client.post("/api/predict", json={"features": {"word_count": "x"}}).status_code, 400)
        self.assertEqual(self.client.post("/api/predict", json={"text": "a" * 20001}).status_code, 413)


if __name__ == "__main__":
    unittest.main()
