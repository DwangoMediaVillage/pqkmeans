# -*- coding:utf-8
import unittest

class TestSample(unittest.TestCase):
    def test_sample(self):
        self.assertEqual(1, 1)

def suite():
    suite = unittest.TestSuite()
    suite.addTests(unittest.makeSuite(TestSample))
    return suite