class Logger:
    def print(self, *args, **kwargs):
        print(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        self.print(*args, **kwargs)