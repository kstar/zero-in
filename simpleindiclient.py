import PyIndi
import time
import sys
import threading

class IndiClient(PyIndi.BaseClient):
    def __init__(self, blobEvent):
        super(IndiClient, self).__init__()
        self._blobEvent = blobEvent

    def newDevice(self, d):
        pass
    def newProperty(self, p):
        pass
    def removeProperty(self, p):
        pass
    def newBLOB(self, bp):
        self._blobEvent.set()
        pass
    def newSwitch(self, svp):
        pass
    def newNumber(self, nvp):
        pass
    def newText(self, tvp):
        pass
    def newLight(self, lvp):
        pass
    def newMessage(self, d, m):
        pass
    def serverConnected(self):
        pass
    def serverDisconnected(self, code):
        pass


def indi_connect(indiclient, server_host="localhost", server_port=7624):
    assert isinstance(indiclient, IndiClient)
    indiclient.setServer(server_host, server_port)
    if not indiclient.connectServer():
        raise RuntimeError('No indiserver running on {}:{}!'.format(indiclient.getHost(), indiclient.getPort()))

def indi_wait(func, timeout=15, poll_interval=0.5):
    time_elapsed = 0.0
    result = func()
    while not result and time_elapsed < timeout:
        time.sleep(poll_interval)
        time_elapsed += poll_interval
        result = func()

    if not result:
        raise TimeoutError('Timed out!')

    return result
