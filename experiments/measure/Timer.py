import time


class Timer:
    """
    A basic timer to track time in experiments
    All times are given in seconds.
    """

    def __init__(self):
        self._time_points = []
        self.paused = False
        self.stopped = False

    def start(self):
        """
        Starts the timer.
        """
        t = time.time()

        if self.stopped:
            raise Exception('Timer is already stopped')

        self._time_points.append(t)

    def stop(self):
        """
        Stops the timer.
        """
        t = time.time()

        if self.stopped:
            raise Exception('Timer is already stopped')

        self._time_points.append(t)
        self.stopped = True

    def pause(self):
        """
        Pauses the timer.
        """
        t = time.time()

        if self.paused:
            raise Exception('Timer is already paused')
        elif self.stopped:
            raise Exception('Timer is already stopped')

        self._time_points.append(t)
        self.paused = True

    def resume(self):
        """
        Resumes the timer.
        """
        t = time.time()

        if not self.paused:
            raise Exception('Timer is not paused')
        elif self.stopped:
            raise Exception('Timer is already stopped')

        self._time_points.append(t)
        self.paused = False

    def time_elapsed(self) -> float:
        """
        Calculates the time that has elapsed between start and stop excluding the pauses.
        :return: The time that has elapsed between start and stop excluding the pauses in seconds.
        """
        if not self.stopped:
            raise Exception('Timer must be stopped.')

        elapsed = 0
        for i in range(0, len(self._time_points), 2):
            start = self._time_points[i]
            stop = self._time_points[i+1]
            elapsed += stop - start

        return elapsed




