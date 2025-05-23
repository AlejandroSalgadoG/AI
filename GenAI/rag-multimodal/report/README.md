
# Multimodal Rag exercise

This is the report for the **Test Task: Building a Multimodal RAG System**.  This report is divided in 5 parts

* Models used
* Web scraper
* Storage
* Retrival and generation
* Testing

## Models used

It was decided that in order to achieve the full development of the project within the stablished time limit this project would be built using the **Ollama** open-source platform to facilitate the running of LLMs in the local environment and specially avoid incurring in any monetary expenses.

* **llama3.2** for embedding
* **llava:13b** for llava:13b
* **gemma3** for multimodal execution
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
    url: str
    label: str | None
    format: str

@dataclass
class  WebData:
    text: str
    metadata: dict[str, str]
    images: list[WebImage]
```

The images downloaded are transformed into a **base 64** str to obtain a compact and easy representation.

## Storage

### sumarization

The text data extracted is divided into chunks using the **RecursiveCharacterTextSplitter** from **langchain_text_splitters** with the separators  **'.'** and **'\n'** and a chunk_size of 1000 and an overlap of 200. On the other hand, the images in **base 64** format were summarized by the llava model.

### Storage

For the storage a **MultiVectorRetriever** from **langchain.retrievers.multi_vector** was used to combine a document and vector storage. **Chroma** was used to store the vector representations and a **LocalFileStorage** from **langchain.storage** was used to store the documents.

The text chunks are stored in both the vector space and the local storage while the in the case of the images the summary stored in the vector store but the base 64 content is stored in the local storage.

## Retrival and generation

For the retrival a **lanchain** chain was implemented. This chain consist of 3 steps

* Prepare prompt
* Multimodal model execution
* Parse output with **StrOutputParser** from **langchain_core.output_parsers**

The preparation of the prompt is composed of three functions. The first stores the context retrieved for testing purposes, the second splits the contents between images and texts, and finally, the third actually creates the prompt using the instructions for the multimodal rag alongside the images and text as context.

## Testing

The implementation present in test_rag.py consists on an execution over a golden dataset using **deepeval**.
The following metrics were used

*  Answer relevancy
*  Faithfulness
*  Contextual precision
*  Contextual recall
*  Contextual relevancy
