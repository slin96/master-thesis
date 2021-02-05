import unittest
from time import sleep

from experiments.measure.eventtimer import EventTimer

TEST_EVENT = 'test_event'
TEST_EVENT_2 = 'test_event_2'
POINT_EVENT = 'point_event'


class TestCopiedModels(unittest.TestCase):

    def test_event_start_stop(self):
        timer = EventTimer()

        elapsed_time = 2.0

        timer.start_event(TEST_EVENT)
        sleep(elapsed_time)
        timer.stop_event(TEST_EVENT)

        measure = timer.time_elapsed(TEST_EVENT)

        self.assertAlmostEqual(elapsed_time, measure, places=2)

    def test_two_events_start_stop(self):
        timer = EventTimer()

        timer.start_event(TEST_EVENT)
        sleep(2)
        timer.start_event(TEST_EVENT_2)
        sleep(1)
        timer.stop_event(TEST_EVENT)
        sleep(3)
        timer.stop_event(TEST_EVENT_2)

        measure_e_1 = timer.time_elapsed(TEST_EVENT)
        measure_e_2 = timer.time_elapsed(TEST_EVENT_2)

        self.assertAlmostEqual(3, measure_e_1, places=2)
        self.assertAlmostEqual(4, measure_e_2, places=2)

    def test_stop_non_existing_event(self):
        with self.assertRaises(Exception):
            timer = EventTimer()

            timer.start_event(TEST_EVENT)
            sleep(2)
            timer.stop_event(TEST_EVENT)
            timer.start_event(TEST_EVENT)

    def test_timeline(self):
        timer = EventTimer()

        timer.start_event(TEST_EVENT)
        sleep(2)
        timer.start_event(TEST_EVENT_2)
        sleep(1)
        timer.stop_event(TEST_EVENT)
        sleep(1)
        timer.point_event(POINT_EVENT)
        sleep(3)
        timer.stop_event(TEST_EVENT_2)

        timeline = timer.get_time_line()
        self.assertEqual(5, len(timeline))

        expected_event_order = [
            TEST_EVENT + '-start',
            TEST_EVENT_2 + '-start',
            TEST_EVENT + '-stop',
            POINT_EVENT,
            TEST_EVENT_2 + '-stop'
        ]

        for expected, actual in zip(expected_event_order, timeline):
            self.assertEqual(expected, actual[0])
