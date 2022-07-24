import pickle


class PipeCommunicator:
    READY_MSG = "ready"
    SUCCEED_MSG = "succeed"
    FAILURE_MSG = "fail"
    EXIT_MSG = "exit"

    def __init__(self, inpipe, outpipe, error_handler=None):
        self.inpipe = inpipe
        self.outpipe = outpipe
        self.error_handler = error_handler

    def recvObject(self, signal=True):
        obj = pickle.load(self.inpipe)
        if signal:
            self.signalSucceed()
        return obj

    def sendObject(self, obj, check=True):
        pickle.dump(obj, self.outpipe)
        self.outpipe.flush()
        if check:
            self.checkSucceed()

    def signalReady(self):
        self.sendObject(self.READY_MSG, check=False)

    def signalSucceed(self):
        self.sendObject(self.SUCCEED_MSG, check=False)

    def signalFail(self):
        self.sendObject(self.FAILURE_MSG, check=False)

    def signalExit(self):
        self.sendObject(self.EXIT_MSG, check=False)

    def checkReady(self):
        obj = self.recvObject(signal=False)
        if obj != self.READY_MSG:
            if self.error_handler:
                self.error_handler()
            raise RuntimeError(f"Received unexpected message {obj}")

    def checkSucceed(self):
        obj = self.recvObject(signal=False)
        if obj != self.SUCCEED_MSG:
            if self.error_handler:
                self.error_handler()
            raise RuntimeError(f"Received unexpected message {obj}")
