import asyncio

from HolyImageDownloader import ImageDownloader


async def main():
    google = ImageDownloader()
    await google.download("stock bird photo", path="raw_images/bird", max_images=250)


asyncio.run(main())
