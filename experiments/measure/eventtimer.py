import time


class StartStopEvent:

    def __init__(self, start):
        self.start = start
        self.stop = None


class PointEvent:
    def __init__(self, time):
        self.time = time


class EventTimer:
    """
    A basic timer to track events
    All times are given in seconds.
    """

    def __init__(self):
        self.events = {}

    def start_event(self, event_name: str):
        """
        Starts a timer for an event.
        :param event_name: The name of the event to time
        """
        if event_name in self.events:
            raise Exception('event \'{}\' already exists'.format(event_name))

        t = time.time()
        event = StartStopEvent(start=t)
        self.events[event_name] = event

    def stop_event(self, event_name: str):
        """
        Stops a timer for an event.
        :param event_name: The name of the event to time
        """
        if event_name not in self.events:
            raise Exception('event \'{}\' does not exist'.format(event_name))

        t = time.time()
        self.events[event_name].stop = t

    def time_elapsed(self, event_name: str) -> float:
        """
        Calculates the time that has elapsed between start and stop of an event.
        :return: The time that has elapsed between start and stop in seconds.
        """
        event = self.events[event_name]
        elapsed = event.stop - event.start

        return elapsed

    def point_event(self, event_name: str):
        """
        Created a point event with a given name.
        :param event_name: The name for the point event.
        """
        if event_name in self.events:
            raise Exception('event \'{}\' already exists'.format(event_name))

        self.events[event_name] = PointEvent(time.time())

    def get_time_line(self, start_with_zero=True) -> [(str, float)]:
        """
        Creates a timeline sorted by time of the event
        :param start_with_zero: If True the first event in the timeline is zeroed an all other events are calculated
        relative to the first event.
        :return: All events sorted by time. A point in time is represented as ("<Event>-<START/STOP>", <TIME>)
        """
        time_line = []

        for event_name, event in self.events.items():
            if isinstance(event, StartStopEvent):
                start = (event_name + '-start', event.start)
                time_line.append(start)
                stop = (event_name + '-stop', event.stop)
                time_line.append(stop)
            else:
                assert isinstance(event, PointEvent)
                time_point = (event_name, event.time)
                time_line.append(time_point)

        # sort by times
        sorted_times = sorted(time_line, key=lambda x: x[1])

        if start_with_zero:
            first_time = sorted_times[0][1]
            normalized = list(map(lambda x: (x[0], x[1] - first_time), sorted_times))
            return normalized
        else:
            return sorted_times

    def get_elapsed_times(self) -> [(str, float)]:
        """
        For all StartStopEvents calculate the elapsed time and return it in a list.
        :return: The time elapsed for all StartStopEvents in seconds.
        """
        result = []

        for event_name, event in self.events.items():
            if isinstance(event, StartStopEvent):
                elapsed = self.time_elapsed(event_name)
                result.append((event_name, elapsed))

        return result
