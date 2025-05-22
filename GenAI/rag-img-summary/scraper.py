import re
import requests
import uuid
import warnings

from collections.abc import Iterator
from datetime import timedelta, date
from functools import reduce
from requests.models import Response
from urllib.parse import urljoin as urljoin_bin, urlsplit

from bs4 import BeautifulSoup
from bs4.element import Tag
from langchain_community.document_loaders import WebBaseLoader

from definitions import WebData
from logs import logger


def iter_dates(start_date: date, delta: timedelta, n: int) -> list[date]:
    return [start_date - delta * i for i in range(n)]


def urljoin(*args: str) -> str:
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


def find_all_tags(soup: BeautifulSoup, tag: str) -> list[Tag]:
    return [t for t in soup.find_all(tag) if isinstance(t, Tag)]


def find_tag(soup: BeautifulSoup, tag: str, **kwargs) -> Tag | None:
    t = soup.find(tag, **kwargs)
    return t if isinstance(t, Tag) else None


def find_within_tag(tag: Tag, tag_name: str, **kwargs) -> Tag | None:
    sub_tag = tag.find(tag_name, **kwargs)
    return sub_tag if isinstance(sub_tag, Tag) else None


def find_all_within_tag(tag: Tag, attr: str, **kwargs) -> list[Tag]:
    return [t for t in tag.find_all(attr, **kwargs) if isinstance(t, Tag)]


def get_str_from_tag_or_none(tag: Tag, attr: str) -> str | None:
    value = tag.get(attr)
    return value if isinstance(value, str) else None


def get_str_from_tag(tag: Tag, attr: str, default: str) -> str:
    value = get_str_from_tag_or_none(tag, attr)
    return default if value is None else value


def get_articles_url(date: date) -> list[str]:
    base_url = "https://www.deeplearning.ai"

    date_str = date.strftime("%b-%d-%Y").lower()
    url = urljoin(base_url, "the-batch/tag/", date_str)

    logger.info(f"Starting article retrival for date {date_str}")

    urls = []
    if response := get_request(url):
        logger.info(f"Index acquired {url}")

        bs = BeautifulSoup(response.content, "html.parser")
        href_regex = re.compile("/the-batch/(?!tag|issue)")

        for article in find_all_tags(bs, "article"):
            if tag := find_within_tag(article, "a", href=href_regex):
                if href := get_str_from_tag_or_none(tag, "href"):
                    url = urljoin(base_url, href)
                    logger.info(f"Article found at {url}")
                    urls.append(url)

    if not urls:
        logger.info("No articles found")

    return urls


class WebLoader(WebBaseLoader):
    def build_metadata(self, soup: BeautifulSoup, url: str) -> dict[str, str]:
        metadata = {"source": url, "uuid": str(uuid.uuid4())}

        if title := find_tag(soup, "title"):
            metadata["title"] = title.get_text()

        if html := find_tag(soup, "html"):
            metadata["language"] = get_str_from_tag(html, "lang", "No language found.")

        if description := find_tag(soup, "meta", attrs={"name": "description"}):
            metadata["description"] = get_str_from_tag(
                description, "content", "No description found."
            )

        return metadata

    def get_text(self, soup: BeautifulSoup) -> str | None:
        article = find_tag(soup, "article")
        if article is None:
            return None

        data = [p.get_text() for p in find_all_within_tag(article, "p")]
        if not data:
            return None

        logger.info(f"Found {len(data)} paragraphs")
        return "\n".join(data)

    def load_web_path(self, web_path: str) -> WebData | None:
        logger.info(f"Starting scraping of {web_path}")
        soup = self._scrape(web_path, bs_kwargs=self.bs_kwargs)

        data = self.get_text(soup)
        if data is None:
            logger.info("Unable to find paragraphs")
            return None

        return WebData(
            text=data,
            metadata=self.build_metadata(soup, web_path),
        )

    def lazy_load_web_paths(self) -> Iterator[WebData]:
        for web_path in self.web_paths:
            if web_data := self.load_web_path(web_path):
                yield web_data
            else:
                logger.info(f"Skiping {web_path}")

    def load_web_paths(self) -> list[WebData]:
        return list(self.lazy_load_web_paths())


if __name__ == "__main__":
    dates = iter_dates(start_date=date(2025, 2, 26), delta=timedelta(days=7), n=1)

    data = []
    for d in dates:
        if articles_url := get_articles_url(d):
            loader = WebLoader(web_paths=articles_url)
            data += loader.load_web_paths()
