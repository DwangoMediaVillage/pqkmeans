import unittest
import pqkmeans


class TestBKMeans(unittest.TestCase):
    def test_just_constuction(self):
        bkmeans = pqkmeans.clustering.BKMeans(32, 2)

    def test_invalid_construction(self):
        self.assertRaises(Exception, lambda : pqkmeans.clustering.BKMeans(10000, 1))
        self.assertRaises(Exception, lambda : pqkmeans.clustering.BKMeans(32, 100))