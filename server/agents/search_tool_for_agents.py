# Beware when updating pydantic to v2, you might need to use pydantic.v1 instead of pydantic namespace
from pydantic import BaseModel, Field
from typing import Any, List, Literal
from bs4 import BeautifulSoup
import requests
from server_config import serper_api_key
from Modules.Summarizer import Summarizer
from langchain.tools import Tool
from helpers.time_function_decorator import time_function
import asyncio
import aiohttp


# ban some sites that we can never scrape
banned_sites = ["calendar.google.com", "researchgate.net"]
# custom search tool, we copied the serper integration on langchain but we prefer all the data to be displayed in one json message
k: int = 3
gl: str = "us"
hl: str = "en"
tbs = None
num_sentences = 5
search_type: Literal["news", "search", "places", "images"] = "search"
summarizer = Summarizer(None)
result_key_for_type = {
        "news": "news",
        "places": "places",
        "images": "images",
        "search": "organic",
    }


# Sync version
@time_function()
def scrape_page(url: str):
    """
    Based on your observations from the Search_Engine, if you want more details from
    a snippet for a non-PDF page, pass this the page's URL and the page's title to
    scrape the full page and retrieve the full contents of the page.
    """

    print("Parsing: {}".format(url))
    if any(substring in url for substring in banned_sites):
        print("Skipping site: {}".format(url))
        return None
    else:
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.84 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
                'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
                'Accept-Encoding': 'none',
                'Accept-Language': 'en-US,en;q=0.8',
                'Connection': 'keep-alive',
            }
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')
            text = " ".join([t.get_text() for t in soup.find_all(
                ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])])
            return text.replace('|', '')
        except requests.RequestException as e:
            print(f"Failed to fetch {url}. Error: {e}")
            return None


@time_function()
def serper_search(
    search_term: str, search_type: str = "search", **kwargs: Any
) -> dict:
    headers = {
        "X-API-KEY": serper_api_key or "",
        "Content-Type": "application/json",
    }
    params = {
        "q": search_term,
        **{key: value for key, value in kwargs.items() if value is not None},
    }
    response = requests.post(
        f"https://google.serper.dev/{search_type}", headers=headers, params=params
    )
    response.raise_for_status()
    search_results = response.json()
    return search_results


@time_function()
def parse_snippets(results: dict) -> List[str]:
    snippets = []
    if results.get("answerBox"):
        answer_box = results.get("answerBox", {})
        if answer_box.get("answer"):
            snippets.append(answer_box.get("answer"))
        elif answer_box.get("snippet"):
            snippets.append(answer_box.get("snippet").replace("\n", " "))
        elif answer_box.get("snippetHighlighted"):
            snippets.append(answer_box.get("snippetHighlighted"))

    if results.get("knowledgeGraph"):
        kg = results.get("knowledgeGraph", {})
        title = kg.get("title")
        entity_type = kg.get("type")
        if entity_type:
            snippets.append(f"{title}: {entity_type}.")
        description = kg.get("description")
        if description:
            snippets.append(description)
        for attribute, value in kg.get("attributes", {}).items():
            snippets.append(f"{title} {attribute}: {value}.")

    for result in results[result_key_for_type[search_type]][:k]:
        if "snippet" in result:
            page = scrape_page(result["link"])
            if ('title' not in result) or (result['title'] is None):
                result['title'] = ""
            if page is None:
                snippets.append(
                    f"Title: {result['title']}\nPossible answers: {result['snippet']}\n")
            else:
                summarized_page = summarizer.summarize_description_with_bert(
                    page, num_sentences=num_sentences)
                if len(summarized_page) == 0:
                    summarized_page = "None"
                snippets.append(
                    f"Title: {result['title']}\nSource:{result['link']}\nSnippet: {result['snippet']}\nSummarized Page: {summarized_page}")

    if len(snippets) == 0:
        return ["No good Google Search Result was found"]
    return snippets


@time_function()
def parse_results(results: dict) -> str:
    snippets = parse_snippets(results)
    results_string = ""
    for idx, val in enumerate(snippets):
        results_string += f"<result{idx}>\n{val}\n</result{idx}>\n\n"
    return results_string


@time_function()
def run_search_tool_for_agents(query: str):
    results = serper_search(
        search_term=query,
        gl=gl,
        hl=hl,
        num=k,
        tbs=tbs,
        search_type=search_type,
    )
    return parse_results(results)


# Async version
@time_function()
async def scrape_page_async(url: str, summarize_page = False, num_sentences = 3):
    """
    Based on your observations from the Search_Engine, if you want more details from
    a snippet for a non-PDF page, pass this the page's URL and the page's title to
    scrape the full page and retrieve the full contents of the page.
    """

    print("Parsing: {}".format(url))
    if any(substring in url for substring in banned_sites):
        print("Skipping site: {}".format(url))
        return None
    else:
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.84 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
                'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
                'Accept-Encoding': 'none',
                'Accept-Language': 'en-US,en;q=0.8',
                'Connection': 'keep-alive',
            }
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, timeout=30) as response:
                    response.raise_for_status()
                    content = await response.text()

                    soup = BeautifulSoup(content, 'html.parser')
                    text = " ".join([t.get_text() for t in soup.find_all(
                        ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])])
                    text = text.replace('|', '')

                    if summarize_page:
                        return summarizer.summarize_description_with_bert(text, num_sentences=num_sentences)
                    return text
        except Exception as e:
            print(f"Failed to fetch {url}. Error: {e}")
            return None


@time_function()
async def serper_search_async(
    search_term: str, search_type: str = "search", **kwargs: Any
) -> dict:
    headers = {
        "X-API-KEY": serper_api_key or "",
        "Content-Type": "application/json",
    }
    params = {
        "q": search_term,
        **{key: value for key, value in kwargs.items() if value is not None},
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(f"https://google.serper.dev/{search_type}", headers=headers, json=params) as response:
            response.raise_for_status()
            search_results = await response.json()
            return search_results


@time_function()
async def parse_snippets_async(results: dict, scrape_pages: bool = False, summarize_pages: bool = True, num_sentences: int = 3) -> List[str]:
    snippets = []
    if results.get("answerBox"):
        answer_box = results.get("answerBox", {})
        if answer_box.get("answer"):
            snippets.append(f"The answer is {answer_box.get('answer')}")
        elif answer_box.get("snippet"):
            snippets.append(f"The answer might be in the snippet: {answer_box.get('snippet')}")
        elif answer_box.get("snippetHighlighted"):
            snippets.append(f"The answer might be in the snippet: {answer_box.get('snippetHighlighted')}")

    if results.get("knowledgeGraph"):
        kg = results.get("knowledgeGraph", {})
        title = kg.get("title")
        entity_type = kg.get("type")
        if entity_type:
            snippets.append(f"Knowledge Graph Results: {title}: {entity_type}.")
        description = kg.get("description")
        if description:
            snippets.append(f"Knowledge Graph Results: {title}: {description}.")
        for attribute, value in kg.get("attributes", {}).items():
            snippets.append(f"Knowledge Graph Results: {title} {attribute}: {value}.")

    if scrape_pages:
        tasks = []
        for result in results[result_key_for_type[search_type]][:k]:
            task = asyncio.create_task(scrape_page_async(result["link"], summarize_page=summarize_pages, num_sentences=num_sentences))
            tasks.append(task)
        summarized_pages = await asyncio.gather(*tasks)
        for i, page in enumerate(summarized_pages):
            result = results[result_key_for_type[search_type]][i]
            if page:
                snippets.append(f"Title: {result.get('title', '')}\nSource:{result['link']}\nSnippet: {result.get('snippet', '')}\nSummarized Page: {page}")
            else:
                snippets.append(f"Title: {result.get('title', '')}\nSource:{result['link']}\nSnippet: {result.get('snippet', '')}\n")
    else:
        for result in results[result_key_for_type[search_type]][:k]:
            snippets.append(f"Title: {result.get('title', '')}\nSource:{result['link']}\nSnippet: {result.get('snippet', '')}\n")

    if len(snippets) == 0:
        return ["No good Google Search Result was found"]
    return snippets


@time_function()
async def parse_results_async(results: dict, scrape_pages: bool = False, summarize_pages: bool = True, num_sentences: int = 3) -> str:
    snippets = await parse_snippets_async(results, scrape_pages=scrape_pages, summarize_pages=summarize_pages, num_sentences=num_sentences)
    results_string = ""
    for idx, val in enumerate(snippets):
        results_string += f"Res {idx + 1}:\n{val}\n\n"
    return results_string


@time_function()
async def arun_search_tool_for_agents(query: str):
    results = await serper_search_async(
        search_term=query,
        gl=gl,
        hl=hl,
        num=k,
        tbs=tbs,
        search_type=search_type,
    )
    return await parse_results_async(results, scrape_pages=False)


class SearchInput(BaseModel):
    query: str = Field(description="a search query")


def get_search_tool_for_agents():
    search_tool_for_agents = Tool(
        name="Search_Engine",
        func=run_search_tool_for_agents,
        coroutine=arun_search_tool_for_agents,
        description="Pass this specific queries and/or keywords to quickly search the WWW to retrieve information on any topic like academic research, history, entertainment, current events. This tool does NOT work for personal information and does NOT work for math.",
        args_schema=SearchInput
    )
    return search_tool_for_agents

from google.cloud import enterpriseknowledgegraph as ekg
from server_config import gcp_project_id
from collections.abc import Sequence

location = 'global'      # Values: 'global'
languages = ['en']                    # Optional: List of ISO 639-1 Codes
types = ['']                          # Optional: List of schema.org types to return
limit = 1                            # Optional: Number of entities to return


# Google knowledge graph search tool
def search_google_knowledge_graph(
    search_query: str,
    project_id: str = gcp_project_id,
    location: str = location, 
    languages: Sequence[str] = languages,
    types: Sequence[str] = None,
    limit: int = limit,):
    
    # Create a client
    client = ekg.EnterpriseKnowledgeGraphServiceAsyncClient()

    # The full resource name of the location
    # e.g. projects/{project_id}/locations/{location}
    parent = client.common_location_path(project=project_id, location=location)

    # Initialize request argument(s)
    request = ekg.SearchPublicKgRequest(
        parent=parent,
        query=search_query,
        languages=languages,
        types=types,
        limit=limit,
    )

    # Make the request
    response = client.search_public_kg(request=request)

    return response


async def asearch_google_knowledge_graph(
    search_query: str,
    project_id: str = gcp_project_id,
    location: str = location, 
    languages: Sequence[str] = languages,
    types: Sequence[str] = None,
    limit: int = limit,):
    
    # Create a client
    client = ekg.EnterpriseKnowledgeGraphServiceAsyncClient()

    # The full resource name of the location
    # e.g. projects/{project_id}/locations/{location}
    parent = client.common_location_path(project=project_id, location=location)

    # Initialize request argument(s)
    request = ekg.SearchPublicKgRequest(
        parent=parent,
        query=search_query,
        languages=languages,
        types=types,
        limit=limit,
    )

    try:
        # Make the request
        response = await client.search_public_kg(request=request)

        # print("response: {}".format(response))
    except:
        print("error in search_google_knowledge_graph")
        response = None
    
    return response

import requests

def can_embed_url(url: str):
    response = requests.head(url)

    # Check the headers for 'X-Frame-Options' or 'Content-Security-Policy'
    x_frame_options = response.headers.get('X-Frame-Options')
    csp = response.headers.get('Content-Security-Policy')

    return not (x_frame_options or ('frame-ancestors' in csp if csp else False))

# definer url search tool
def extract_entity_url_and_image(search_results: dict, image_results: dict):
    # Only get the first top url and image_url
    res = {}
    if search_results.get("knowledgeGraph"):
        result = search_results.get("knowledgeGraph", {})
        if result.get("descriptionSource") == "Wikipedia":
            ref_url = result.get("descriptionLink")
            res["url"] = ref_url

    for result in search_results[result_key_for_type["search"]][:k]:
        if "url" not in res and result.get("link") and can_embed_url(result.get("link")):
            res["url"] = result.get("link")
            break

    if image_results is None:
        return res
    
    for result in image_results[result_key_for_type["images"]][:k]:
        if "image_url" not in res and result.get("imageUrl"):
            res["image_url"] = result.get("imageUrl")
            break

    return res

async def search_url_for_entity_async(query: str, get_image: bool = True):
    async def inner_search(query:str): 
        search_task = asyncio.create_task(serper_search_async(
            search_term=query,
            gl=gl,
            hl=hl,
            num=k,
            tbs=tbs,
            search_type="search",
        ))

        if get_image:
            image_search_task = asyncio.create_task(serper_search_async(
                search_term=query,
                gl=gl,
                hl=hl,
                num=k,
                tbs=tbs,
                search_type="images",
            ))
        else:
            async def dummy(): return None
            image_search_task = asyncio.create_task(dummy())

        tasks = [search_task, image_search_task]

        search_results, image_results = await asyncio.gather(*tasks)
        
        return extract_entity_url_and_image(search_results, image_results)
    
    res = await inner_search(query)
    if "url" not in res:
        res = await inner_search(query + " wiki") # fallback search using wiki
    return res
