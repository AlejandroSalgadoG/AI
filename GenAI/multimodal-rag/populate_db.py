from argparse import ArgumentParser, ArgumentTypeError
from datetime import date, datetime, timedelta

from storage import Storage
from scraper import iter_dates, get_articles_url, MultimodalWebLoader
from summary_image import ImageSummarizer
from summary_text import TextSummarizer


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
        description="script to populate database for multimodal rag example",
    )

    parser.add_argument(
        "start_date",
        help="wednesday date from when to start scraping. Format dd-mm-yyyy",
        type=valid_start_date,
    )
    parser.add_argument(
        "num_issues",
        help="number of batchs to scrape starting from <start_date> and going backwards. Int",
        type=valid_num_issues,
    )
    return parser


if __name__ == '__main__':
    arg_parser = get_arg_parser()
    args = arg_parser.parse_args()

    storage = Storage()
    text_summarizer = TextSummarizer()
    image_summarizer = ImageSummarizer()
    dates = iter_dates(start_date=args.start_date, delta=timedelta(days=7), n=args.num_issues)

    for d in dates:
        if articles_url := get_articles_url(d):
            loader = MultimodalWebLoader(web_paths=articles_url)
            for web_data in loader.lazy_load_web_paths():
                text_documents = text_summarizer.apply(web_data)
                storage.add_vector_and_doc_info(
                    vector_store_docs=text_documents.get_summary_docs(),
                    doc_store_docs=text_documents.get_content_docs(),
                )

                image_documents = image_summarizer.apply(web_data)
                storage.add_vector_and_doc_info(
                    vector_store_docs=image_documents.get_summary_docs(),
                    doc_store_docs=image_documents.get_summary_docs(),
                    # the image contents can also be stored in the doc_store if desired
                )
