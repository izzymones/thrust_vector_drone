# This is our Interrupt timer from the balancing robot. We might want to upgrade 
# it to something like a GridTimer. Something that would manage all our scheduling
# and keep things synched. Maybe we add some guidance algorithm later that runs 
# at a lower frequency but still needs to be synched. What would that look like?

from threading import Timer
from time import perf_counter, sleep

class InterruptTimer:
    def __init__(self, dt, callback_function, timeout):
        self.dt = dt
        self.callback_function = callback_function
        self.timeout = perf_counter() + timeout
        self.running = False
  
    def work(self, target_time):
        t0 = perf_counter()
        self.callback_function()
        if self.timeout < t0:
            self.running = False
            return
        else: 
            Timer(target_time - t0, self.work, args=[target_time + self.dt]).start()

    def start(self):
        self.running = True
        target_time = perf_counter() + self.dt
        self.work(target_time)


########### test timer ############


if __name__ == "__main__":
    def my_callback():
        print("called callback")

    timer = InterruptTimer(4, my_callback, 5)
    timer.start()
    while timer.running:
        print("active")
        sleep(0.2)
    print("timer timed out")

