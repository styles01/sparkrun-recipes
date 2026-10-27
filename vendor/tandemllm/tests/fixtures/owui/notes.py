"""
title: Word Tools
description: Count and transform text.
"""


class Tools:
    class Valves:
        def __init__(self, **kw):
            self.UPPER = kw.get("UPPER", False)

        UPPER: bool = False

    def __init__(self):
        self.valves = self.Valves()

    def word_count(self, text: str, unique: bool = False) -> dict:
        """Count the words in a text.

        :param text: The text.
        :param unique: Count each word once.
        """
        words = text.split()
        return {"words": len(set(words)) if unique else len(words)}

    async def shout(self, text: str, __user__: dict = None, __metadata__: dict = None) -> str:
        """Repeat the text, upper case when the UPPER valve is on."""
        print("shouting")
        return text.upper() if self.valves.UPPER else text + "!"

    def tags(self, items: list[str], sep: str | None = None) -> str:
        """Join tags."""
        return (sep or ", ").join(items)
