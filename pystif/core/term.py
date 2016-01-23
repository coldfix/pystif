
import os
import termios
import select

from collections import namedtuple

Termios = namedtuple('Termios', [
    'iflag',
    'oflag',
    'cflag',
    'lflag',
    'ispeed',
    'ospeed',
    'cc',
])


class TerminalInput:

    def __init__(self):
        self.fd = os.open("/dev/tty", os.O_RDONLY)
        # Backup and modify TTY state:
        self._backup = tios = Termios(*termios.tcgetattr(self.fd))
        tios = tios._replace(
            lflag=tios.lflag & ~(termios.ICANON | termios.ECHO),
            cc=tios.cc[:])
        tios.cc[termios.VMIN] = 1   # minimum of number input read.
        termios.tcsetattr(self.fd, termios.TCSANOW, list(tios))

    def close(self):
        if self.fd is None:
            return
        termios.tcsetattr(self.fd, termios.TCSANOW, list(self._backup))
        os.close(self.fd)
        self.fd = None

    def __del__(self): self.close()
    def __enter__(self): return self
    def __exit__(self, *exc_info): self.close()

    def get(self):
        return os.read(self.fd, 1)

    def avail(self):
        # from http://cc.byexamples.com/2007/04/08/non-blocking-user-input-in-loop-without-ncurses/
        rl, wl, xl = select.select([self.fd], [], [], 0)
        return self.fd in rl


if __name__ == '__main__':
    from itertools import count
    from time import sleep
    term = TerminalInput()
    for i in count():
        if term.avail():
            c = term.get()
            print("User input:", c)
            if c == b'q':
                break
        print("Iteration:", i)
        sleep(0.5)
