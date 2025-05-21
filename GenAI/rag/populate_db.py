from argparse import ArgumentParser, ArgumentTypeError
from datetime import date, datetime, timedelta

from storage import Storage
from scraper import iter_dates, get_articles_url, WebLoader
from text_handler import TextHandler


def valid_start_date(start_date: str) -> date:
    try:
        date = datetime.strptime(start_date, "%d-%m-%Y").date()
        if date.weekday() != 2:
            raise ArgumentTypeError(f"not a wednesday date: {start_date}")
        return date
    except ValueError:
        raise ArgumentTypeError(f"not a valid date: {start_date}")


def valid_num_issues(num_issues: str) -> int:
    try:
        return int(num_issues)
    except ValueError:
        raise ArgumentTypeError(f"not a valid number: {num_issues}")


def get_arg_parser():
    parser = ArgumentParser(
        prog="populate_db.py",
        description="script to populate database for rag example",
    )

    parser.add_argument(
        "start_date",
        help="wednesday date from when to start scraping. Format dd-mm-yyyy",
        type=valid_start_date,
        nargs="?",
        default="14-05-2025",
    )

    parser.add_argument(
        "num_issues",
        help="number of batchs to scrape starting from <start_date> and going backwards. Int",
        type=valid_num_issues,
        nargs="?",
        default=1,
    )
    return parser


if __name__ == "__main__":
    arg_parser = get_arg_parser()
    args = arg_parser.parse_args()

    storage = Storage()
    text_handler = TextHandler()
    dates = iter_dates(
        start_date=args.start_date,
        delta=timedelta(days=7),
        n=args.num_issues,
    )

    for d in dates:
        if articles_url := get_articles_url(d):
            loader = WebLoader(web_paths=articles_url)
            for web_data in loader.lazy_load_web_paths():
                docs = text_handler.apply(web_data)
                storage.add_documents(docs)
