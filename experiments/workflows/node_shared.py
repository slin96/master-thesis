import socket

MSG_LEN = 1024


def listen(receiver, callback):
    # socket.SOCK_DGRAM use UDP
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # allow to reuse the socket address
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(receiver)
    received = sock.recvfrom(MSG_LEN)
    callback(received)
    sock.detach()
    sock.close()
