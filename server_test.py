from client_api import AugmoveListener

augmove = AugmoveListener()

while True:
    print(augmove.track_controllers(True))