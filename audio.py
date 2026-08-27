try:
    import pyttsx3
except Exception:
    pyttsx3 = None


class Speaker:
    """
    Text-to-speech helper.

    If pyttsx3 does not work on the computer, the program still continues
    and simply prints messages in the console.
    """

    def __init__(self, rate: int = 150, volume: float = 1.0):
        self.engine = None

        if pyttsx3 is not None:
            try:
                self.engine = pyttsx3.init()
                self.engine.setProperty("rate", rate)
                self.engine.setProperty("volume", volume)
            except Exception:
                self.engine = None

    def say(self, text: str) -> None:
        print(text)

        if self.engine is None:
            return

        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception:
            pass