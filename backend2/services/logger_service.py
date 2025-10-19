from icecream import ic


class LoggerService:
    def __init__(self, filename: str = "log.txt"):
        self.filename = filename
        self.file = open(self.filename, "a")  # noqa: SIM115

        ic.configureOutput(outputFunction=self._write_to_file)

        self.file.write("\n\n--- New Session Started ---\n")

    def _write_to_file(self, message: str):
        self.file.write(message + "\n")
        self.file.flush()

    def log(self, message: str):
        ic(message)

    def __del__(self):
        if self.file:
            self.file.close()
