import unittest
from os import path


if __name__ == '__main__':
    suite = unittest.TestSuite()
    cur_dir = path.dirname(path.abspath(__file__))

    suite.addTests(
        unittest.TestLoader().discover(
            path.join(cur_dir,'model'), 't_*.py', top_level_dir=None))
    suite.addTests(
        unittest.TestLoader().discover(
            path.join(cur_dir,'fquant'), 't_*.py', top_level_dir=None))

    runner = unittest.TextTestRunner(verbosity=1)
    runner.run(suite)
