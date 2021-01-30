import socket

MSG_LEN = 1024


def listen(receiver, callback):
    # socket.SOCK_DGRAM use UDP
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(receiver)
    received = sock.recvfrom(MSG_LEN)
    callback(received)
