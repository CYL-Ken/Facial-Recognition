import time
import requests


class DoorController():
    def __init__(self, wait_time=5) -> None:
        self.checker = ["", "", "", "", ""]
        self.wait_time = wait_time
        self.open_timer = 0
    
    def open(self):
        if (time.time() - self.open_timer) < self.wait_time:
            return
        # response = requests.get(r"http://admin:admin@192.168.50.2/DP/doorunlock.ncgi?id=2635107228")
        print("OPEN!")
        self.open_timer = time.time()
        
    def visit(self, text):
        name = "No Person" if text == None else text
        self.checker.append(name)
        self.checker.pop(0)
        if name != "Guest" and name != "No Person":
            if self.checker.count(name) > 3:
                self.open()