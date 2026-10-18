from urllib.parse import quote

import requests
from mcp.server.fastmcp import FastMCP


mcp = FastMCP("WeatherServer")


@mcp.tool()
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    city = city.strip()

    if not city:
        return "Weather unavailable: please provide a city."

    url = f"https://wttr.in/{quote(city, safe='')}"

    try:
        response = requests.get(
            url,
            params={"format": "j1"},
            timeout=20,
        )
        response.raise_for_status()

        data = response.json()
        current = data["current_condition"][0]

        temperature = current["temp_C"]
        description = current["weatherDesc"][0]["value"]

        return (
            f"Weather in {city}: "
            f"{temperature}°C, {description}."
        )

    except (
        requests.RequestException,
        ValueError,
        KeyError,
        IndexError,
    ) as error:
        return (
            "Weather unavailable: "
            f"{type(error).__name__}: {error}"
        )


if __name__ == "__main__":
    mcp.run(transport="stdio")