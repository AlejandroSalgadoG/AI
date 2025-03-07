
# Multimodal Rag exercise

This is the report for the **Test Task: Building a Multimodal RAG System**.  This report is divided in 5 parts

* Models used
* Web scraper
* Storage
* Retrival and generation
* Testing

## Models used

It was decided that in order to achieve the full development of the project within the stablished time limit this project would be built using the **Ollama** open-source platform to facilitate the running of LLMs in the local environment and specially avoid incurring in any monetary expenses. Due to time constrains the models used were selected based on a quick research and the popularity they displayed in the platform.

* **llama3.1** for text summarization and embedding
* **bakllava** for image summarization
* **llama3.2-vision** for multimodal processing
* **deepseek-r1:1.5b** for testing

## Web scraper

The objective was to download the information available at https://www.deeplearning.ai/the-batch/

It was found that the way they add content was using a release schedule every Wednesday. The first publication was on august 21 of 2019 and the latest publication is march 5 of 2025 (to the date this report is created), resulting in 290 publications so far.

With this information in mind, the first step of the implemented scraper is to download the list of published articles using the url https://www.deeplearning.ai/the-batch/tag/<publication_date>/.

The resulting article urls are used as an input to an extended **WebBaseLoader** from **langchain_community.document_loaders** that contains methods specially design to extract the information from the articles from **The Batch** web page.

The results are stored in the following dataclasses

```
@dataclass
class  WebImage:
    data_b64: str
    metadata: dict[str, str]

@dataclass
class  WebText:
    data: str
    metadata: dict[str, str]

@dataclass
class  WebData:
    text: WebText
    images: list[WebImage]
```

The images downloaded are resized to a standard 680x385 that resemble the size they take when displayed in the web page and later are transformed into a **base 64** str to obtain a compact and easy representation.

## Storage

### sumarization

The text data extracted is divided into chunks using the **RecursiveCharacterTextSplitter** from **langchain_text_splitters** with the separators  **'.'** and **'\n'** and a chunk_size of 2000  (to roughly represent two paragraphs). The results were summarized using the **llama3.1** model. On the other hand, the images in **base 64** format were summarized directly by the bakllava model.

Each summary created was associated with its complete version using a **uuid**.

### Storage

For the storage a **MultiVectorRetriever** from **langchain.retrievers.multi_vector** was used to combine a document and vector storage. **Chroma** was used to store the vector representations and a **LocalFileStorage** from **langchain.storage** was used to store the documents.

The idea is to create a compact and fast vector storage to be used when a query is executed and later the actual complete information that is associated is returned instead of the actual vectors.

In the case of text data the summaries are embedded with llama3.1 and stored in the vector store while the original text content is stored in the doc store. In the case of images the usage of the actual image in the generation part was replaced with the usage of just the summary due to time constrains, however, the code is capable of use the actual images with minor modifications.

## Retrival and generation

For the retrival a **lanchain** chain was implemented. This chain consist of 3 steps

* Prepare prompt
* Multimodal model execution (**llama3.2-vision**)
* Parse output with **StrOutputParser** from **langchain_core.output_parsers**

The preparation of the prompt is composed of three functions. The first stores the context retrieved for testing purposes, the second splits the contents between images and texts, and finally, the third actually creates the prompt using the instructions for the multimodal rag alongside the images and text as context.

## Testing

The testing of the RAG could not be implemented extensively due to time constrains. The implementation present in test_rag.py is a skeleton that can be easily extended to more cases. This  implementation consist of 4 metrics using **deepeval** with deepseek as backend llm

* Correctness using the G-Eval methodology
* Answer relevancy
* Faithfulness
* Context relevancy
