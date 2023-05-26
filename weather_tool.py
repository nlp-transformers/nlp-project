from langchain.utilities import OpenWeatherMapAPIWrapper
from langchain.tools import Tool

weather = OpenWeatherMapAPIWrapper()
weather_tool = Tool.from_function(
        func=weather.run,
        name="weather",
        description="Use this tool only to search for any weather related information, If the user doesn't give a location, use california for a location",
        return_direct=True
    )
