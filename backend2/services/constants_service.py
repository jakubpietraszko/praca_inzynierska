class ConstantsService:

    def __init__(self):
        from constans import Constants
        self.constants = Constants()


    def get_translations(self) -> dict[str, dict[str, str]]:
        return {"translations": self.constants.XAI_TRANSLATIONS}
