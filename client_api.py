import requests

class AugmoveListener:
    def __init__(self, url='http://127.0.0.1:8000/'):
        self.url = url

    def track_controllers(self, show=False):
        return requests.get(self.url.rstrip('/') + '/track-controllers', params={'show': show}).json()