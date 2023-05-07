import unittest
import sys
import keras

sys.path.append("src")
from scripts.train import train_model


class TestTrainer(unittest.TestCase):
    def test_creating_scraper_object(self):
        # when
        model = train_model()
        # then
        self.assertIsInstance(model, keras.engine.functional.Functional)


if __name__ == '__main__':
    unittest.main()
