from queue import Queue, Empty
from threading import Thread, current_thread
import sys


class Printer:
    def __init__(self):
        self.queues = {}

    def write(self, value):
        queue = self.queues.get(current_thread().name)
        if queue:
            queue.put(value)
        else:
            sys.__stdout__.write(value)

    def flush(self):
        pass

    def register(self, thread):
        queue = Queue()
        self.queues[thread.name] = queue
        return queue

    def clean(self, thread):
        del self.queues[thread.name]

printer = Printer()
sys.stdout = printer


class Streamer:
    def __init__(self, target, args):
        self.thread = Thread(target=target, args=args)
        self.queue = printer.register(self.thread)

    def start(self):
        self.thread.start()
        # print('This should be stdout')
        while self.thread.is_alive():
            try:
                item = self.queue.get_nowait()
                yield item.strip()
            except Empty:
                pass
        yield '\n***End***'
        printer.clean(self.thread)
