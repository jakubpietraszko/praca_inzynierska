from icecream import ic

from services.config_service import XAIMethodConfig


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

    def log_config(self, config: XAIMethodConfig, xai_method_name: str):
        self.log(f"{xai_method_name}: {config}")

    def __del__(self):
        if self.file:
            self.file.close()
