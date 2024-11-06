from langchain_community.tools.tavily_search import TavilySearchResults


def get_profile_url_tavily(name: str, mock: bool = True) -> dict[str, str]:
    if mock:
        return {
            'query': name,
            'follow_up_questions': None,
            'answer': None,
            'images': None,
            'results': [
                {
                    'title': 'Eden Marco Udemy - linkedin',
                    'url': '"https://www.linkedin.com/in/eden-marco"',
                    'content': '',
                    'score': 0.98567,
                    'raw_content': None
                }
            ],
            'response_time': 1.07
        }

    search = TavilySearchResults()
    return search.run(name)
