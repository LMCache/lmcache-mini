# SPDX-License-Identifier: Apache-2.0
import itertools
import os
import pickle
import queue
import threading
from concurrent.futures import Future

import zmq

from lmcache_mini.protocol import Req


class SocketLoop:
    def __init__(self, sock):
        self.sock = sock
        self.send_queue = queue.SimpleQueue()
        self.wake_read, self.wake_write = os.pipe()
        self.stopped = False
        self.thread = threading.Thread(target=self.run, daemon=True)

    def start(self):
        self.thread.start()

    def send(self, frames):
        self.send_queue.put(frames)
        os.write(self.wake_write, b"x")

    def close(self):
        self.stopped = True
        os.write(self.wake_write, b"x")
        self.thread.join()
        self.sock.close()
        os.close(self.wake_read)
        os.close(self.wake_write)

    def run(self):
        poller = zmq.Poller()
        poller.register(self.sock, zmq.POLLIN)
        poller.register(self.wake_read, zmq.POLLIN)
        while not self.stopped:
            ready = dict(poller.poll())
            if self.wake_read in ready:
                os.read(self.wake_read, 4096)
                while not self.send_queue.empty():
                    self.sock.send_multipart(self.send_queue.get())
            if self.sock in ready:
                self.on_recv(self.sock.recv_multipart())


class MQClient(SocketLoop):
    def __init__(self, server_url):
        sock = zmq.Context.instance().socket(zmq.DEALER)
        sock.connect(server_url)
        super().__init__(sock)
        self.uid_counter = itertools.count()
        self.pending = {}
        self.start()

    def submit(self, req: Req, payload=None) -> Future:
        uid, fut = next(self.uid_counter), Future()
        self.pending[uid] = fut
        self.send([pickle.dumps((uid, req, payload))])
        return fut

    def call(self, req: Req, payload=None):
        return self.submit(req, payload).result()

    def on_recv(self, frames):
        uid, value = pickle.loads(frames[0])
        self.pending.pop(uid).set_result(value)


class MQServer(SocketLoop):
    def __init__(self, bind_url):
        sock = zmq.Context.instance().socket(zmq.ROUTER)
        sock.bind(bind_url)
        super().__init__(sock)
        self.handlers = {}

    def register(self, req: Req, fn):
        self.handlers[req] = fn

    def on_recv(self, frames):
        identity, blob = frames
        threading.Thread(target=self.handle, args=(identity, blob),
                         daemon=True).start()

    def handle(self, identity, blob):
        uid, req, payload = pickle.loads(blob)
        self.send([identity, pickle.dumps((uid, self.handlers[req](payload)))])
