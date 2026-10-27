"""
title: Weather Lookup
author: tandem-tests
description: Current weather for a city from a weather API.
requirements: requests
version: 1.1
"""

from pydantic import BaseModel, Field


class Tools:
    class Valves(BaseModel):
        API_KEY: str = Field(default="", description="The weather service's API key")
        UNITS: str = Field(default="metric", description="metric or imperial")
        TIMEOUT: int = 10

    def __init__(self):
        self.valves = self.Valves()

    async def get_weather(self, city: str, __event_emitter__=None, __user__: dict = {}) -> str:
        """
        Get the current weather for a city.
        :param city: The city, for example "Berlin".
        """
        import requests
        if __event_emitter__:
            await __event_emitter__({"type": "status", "data": {"description": "Asking", "done": False}})
        r = requests.get("https://api.example.invalid/weather",
                         params={"q": city, "units": self.valves.UNITS, "key": self.valves.API_KEY},
                         timeout=self.valves.TIMEOUT)
        return r.text
