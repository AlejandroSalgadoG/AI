import base64
import io
import re
import requests
import warnings

from collections.abc import Iterator
from datetime import timedelta, date
from functools import reduce
from requests.models import Response
from urllib.parse import urljoin as urljoin_bin, urlsplit

from bs4 import BeautifulSoup
from langchain_community.document_loaders import WebBaseLoader
from PIL import Image

from definitions import WebData, WebImage, WebText


def iter_dates(start_date: date, delta: timedelta, n: int) -> list[date]:
    return [start_date - delta * i for i in range(n)]


def urljoin(*args: list[str]) -> str:
    return reduce(urljoin_bin, args)


def baseurl(url: str) -> str:
    split = urlsplit(url)
    return f"{split.scheme}://{split.netloc}"


def get_request(url: str) -> Response | None:
    response = requests.get(url)
    if response.status_code != 200:
        warnings.warn(f"unable to get response from {url}")
        return None
    return response


def get_articles_url(date: date) -> list[str]:
    base_url = "https://www.deeplearning.ai"

    date_str = date.strftime("%b-%d-%Y").lower()
    url = urljoin(base_url, "the-batch/tag/", date_str)

    urls = []
    if response := get_request(url):
        bs = BeautifulSoup(response.content, "html.parser")
        href_regex = re.compile("/the-batch/(?!tag|issue)")

        for article in bs.find_all("article"):
            if tag := article.find("a", href=href_regex):
                url = urljoin(base_url, tag["href"])
                urls.append(url)

    return urls


def resize_image(image: bytes, size: tuple[int, int]) -> bytes:
    img = Image.open(io.BytesIO(image))
    buffered = io.BytesIO()
    resized_img = img.resize(size, Image.LANCZOS)
    resized_img.save(buffered, format=img.format)
    return buffered.getvalue()


class MultimodalWebLoader(WebBaseLoader):
    def build_metadata(self, soup: BeautifulSoup, url: str) -> dict[str, str]:
        metadata = {"source": url}

        if title := soup.find("title"):
            metadata["title"] = title.get_text()

        if html := soup.find("html"):
            metadata["language"] = html.get("lang", "No language found.")

        # if description := soup.find("meta", attrs={"name": "description"}):
        #     metadata["description"] = description.get("content", "No description found.")

        return metadata

    def get_images(self, soup: BeautifulSoup, url: str) -> list[WebImage]:
        base_url = baseurl(url)
        src_regex = re.compile(f"/_next/image")

        article = soup.find("article")
        if article is None:
            return []

        header = article.find("header")
        if header is None:
            return []

        images = []
        for img in header.find_all("img", src=src_regex):
            image_url = urljoin(base_url, img["src"])
            if response := get_request(image_url):
                image = resize_image(response.content, size=(680, 385))
                images.append(
                    WebImage(
                        data_b64=base64.b64encode(image).decode("utf-8"),
                        metadata={"url": image_url, "description": img.get("alt")},
                    )
                )

        return images

    def get_text(self, soup: BeautifulSoup) -> str | None:
        article = soup.find("article")
        if article is None:
            return None

        data = [p.get_text() for p in article.find_all("p")]
        if not data:
            return None

        return "\n".join(data)

    def load_web_path(self, web_path: str) -> WebData | None:
        soup = self._scrape(web_path, bs_kwargs=self.bs_kwargs)

        data = self.get_text(soup)
        if data is None:
            return None

        return WebData(
            text=WebText(
                data=data,
                metadata=self.build_metadata(soup, web_path),
            ),
            images=self.get_images(soup, web_path),
        )

    def lazy_load_web_paths(self) -> Iterator[WebData]:
        for web_path in self.web_paths:
            if web_data := self.load_web_path(web_path):
                yield web_data

    def load_web_paths(self) -> list[WebData]:
        return list(self.lazy_load_web_paths())


if __name__ == '__main__':
    dates = iter_dates(start_date=date(2025, 2, 26), delta=timedelta(days=7), n=1)

    data = []
    for d in dates:
        if articles_url := get_articles_url(d):
            loader = MultimodalWebLoader(web_paths=articles_url)
            data += loader.load_web_paths()
