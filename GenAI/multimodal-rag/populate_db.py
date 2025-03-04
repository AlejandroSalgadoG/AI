from datetime import timedelta, date

from storage import Storage
from scraper import iter_dates, get_articles_url, MultimodalWebLoader
from summary_image import ImageSummarizer
from summary_text import TextSummarizer


if __name__ == '__main__':
    storage = Storage()
    text_summarizer = TextSummarizer()
    image_summarizer = ImageSummarizer()
    dates = iter_dates(start_date=date(2025, 2, 26), delta=timedelta(days=7), n=288)

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
