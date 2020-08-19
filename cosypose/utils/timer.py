import datetime


class Timer:
    def __init__(self):
        self.start_time = None
        self.elapsed = datetime.timedelta()
        self.is_running = False

    def reset(self):
        self.start_time = None
        self.elapsed = 0.
        self.is_running = False

    def start(self):
        self.elapsed = datetime.timedelta()
        self.is_running = True
        self.start_time = datetime.datetime.now()
        return self

    def pause(self):
        if self.is_running:
            self.elapsed += datetime.datetime.now() - self.start_time
            self.is_running = False

    def resume(self):
        if not self.is_running:
            self.start_time = datetime.datetime.now()
            self.is_running = True

    def stop(self):
        self.pause()
        elapsed = self.elapsed
        self.reset()
        # return elapsed.microseconds / 1000
        return elapsed
