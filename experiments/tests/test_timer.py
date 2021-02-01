import unittest
from time import sleep

from experiments.measure.Timer import Timer


class TestCopiedModels(unittest.TestCase):

    def test_time_start_stop(self):
        timer = Timer()

        elapsed_time = 2.0

        timer.start()
        sleep(elapsed_time)
        timer.stop()

        measure = timer.time_elapsed()

        self.assertAlmostEqual(elapsed_time, measure, places=2)

    def test_time_start_pause_stop(self):
        timer = Timer()

        timer.start()
        sleep(2)
        timer.pause()
        sleep(1)
        timer.resume()
        sleep(3)
        timer.stop()

        measure = timer.time_elapsed()

        self.assertAlmostEqual(5, measure, places=1)

    def test_stop_twice(self):
        with self.assertRaises(Exception):
            timer = Timer()

            timer.start()
            sleep(2)
            timer.stop()
            timer.stop()

    def test_pause_twice(self):
        with self.assertRaises(Exception):
            timer = Timer()

            timer.start()
            sleep(2)
            timer.pause()
            timer.pause()

    def test_resume_no_pause(self):
        with self.assertRaises(Exception):
            timer = Timer()

            timer.start()
            sleep(2)
            timer.resume()

    def test_not_stopped(self):
        with self.assertRaises(Exception):
            timer = Timer()

            timer.start()
            timer.time_elapsed()

    def test_paused_not_stopped(self):
        with self.assertRaises(Exception):
            timer = Timer()

            timer.start()
            sleep(2)
            timer.pause()

            timer.time_elapsed()

