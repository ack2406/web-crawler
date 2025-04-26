import asyncio
import hashlib
import logging
import os
import time
import argparse
from urllib.parse import urlparse, urljoin, urldefrag
from urllib.robotparser import RobotFileParser

import aiohttp
import aiofiles
import networkx as nx
from bs4 import BeautifulSoup

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (compatible; PythonCrawler/1.0; +http://example.com/bot.html)"
)


class AsyncCrawler:
    def __init__(
        self,
        start_url,
        max_pages,
        concurrency,
        output_dir,
        user_agent=DEFAULT_USER_AGENT,
    ):
        self.start_url = start_url
        self.max_pages = max_pages
        self.output_dir = output_dir
        self.html_dir = os.path.join(output_dir, "html_docs")
        self.graph_file = os.path.join(output_dir, "link_graph.graphml")
        self.user_agent = user_agent

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.html_dir, exist_ok=True)
        logging.info(f"Output directory: {self.output_dir}")
        logging.info(f"HTML directory: {self.html_dir}")

        parsed_start_url = urlparse(start_url)
        self.base_domain = parsed_start_url.netloc

        if self.base_domain.count(".") == 1 and not self.base_domain.startswith("www."):
            self.allowed_domains = {self.base_domain, f"www.{self.base_domain}"}
        else:
            self.allowed_domains = {self.base_domain}
        logging.info(f"Base domain: {self.base_domain}")
        logging.info(f"Initial allowed domains: {self.allowed_domains}")

        self.queue = asyncio.Queue()
        self.visited_urls = set()
        self.in_progress_urls = set()
        self.robot_parsers = {}
        self.graph = nx.DiGraph()
        self.page_count = 0

        self.semaphore = asyncio.Semaphore(concurrency)
        self.session = None

    async def _get_robot_parser(self, domain):
        if domain in self.robot_parsers:
            return self.robot_parsers[domain]

        robots_url = f"http://{domain}/robots.txt"
        parser = RobotFileParser()
        parser.set_url(robots_url)
        try:
            async with self.session.get(  # type: ignore
                robots_url, timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    content = await response.text()
                    parser.parse(content.splitlines())
                    logging.info(f"Fetched and parsed robots.txt for {domain}")
                elif response.status == 404:
                    logging.warning(
                        f"robots.txt not found for {domain} (404). Allowing access."
                    )
                    parser.parse([])
                else:
                    logging.warning(
                        f"Failed to fetch robots.txt for {domain} (Status: {response.status}). Allowing access."
                    )
                    parser.parse([])

        except aiohttp.ClientError as e:
            logging.warning(
                f"Network error fetching robots.txt for {domain}: {e}. Allowing access."
            )
            parser.parse([])
        except Exception as e:
            logging.error(
                f"Unexpected error processing robots.txt for {domain}: {e}. Allowing access."
            )
            parser.parse([])

        self.robot_parsers[domain] = parser
        return parser

    async def _can_fetch(self, url):
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        if not domain:
            return False

        try:
            parser = await self._get_robot_parser(domain)
            can = parser.can_fetch(self.user_agent, url)
            if not can:
                logging.debug(f"Access denied by robots.txt: {url}")
            return can
        except Exception as e:
            logging.error(
                f"Error checking robots.txt for {url}: {e}. Cautiously allowing."
            )
            return True

    def _normalize_url(self, base_url, link):
        try:
            joined_url = urljoin(base_url, link)
            url_without_fragment, _ = urldefrag(joined_url)
            return url_without_fragment
        except ValueError:
            logging.debug(f"Failed to normalize link: {link} from base {base_url}")
            return None

    def _is_valid_url(self, url):
        if not url:
            return False
        try:
            parsed_url = urlparse(url)
            if parsed_url.scheme not in ["http", "https"]:
                logging.debug(f"Rejected URL with unsupported scheme: {url}")
                return False

            domain = parsed_url.netloc
            if not domain or not (
                domain == self.base_domain or domain.endswith(f".{self.base_domain}")
            ):
                logging.debug(f"Rejected URL outside allowed domains: {url}")
                return False

            path = parsed_url.path
            if "." in os.path.basename(path):
                extension = os.path.splitext(path)[1].lower()
                allowed_extensions = {"", ".htm", ".html", ".php", ".asp", ".aspx"}
                ignored_extensions = {
                    ".pdf",
                    ".doc",
                    ".docx",
                    ".xls",
                    ".xlsx",
                    ".ppt",
                    ".pptx",
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".gif",
                    ".bmp",
                    ".svg",
                    ".webp",
                    ".zip",
                    ".rar",
                    ".tar",
                    ".gz",
                    ".7z",
                    ".mp3",
                    ".mp4",
                    ".avi",
                    ".mov",
                    ".wmv",
                    ".css",
                    ".js",
                    ".xml",
                    ".json",
                    ".csv",
                    ".txt",
                    ".ics",
                }
                if (
                    extension not in allowed_extensions
                    and extension in ignored_extensions
                ):
                    logging.debug(
                        f"Rejected URL with unsupported file extension: {url}"
                    )
                    return False

            return True
        except ValueError:
            logging.debug(f"Rejected invalid URL: {url}")
            return False

    def _hash_url(self, url):
        return hashlib.sha256(url.encode("utf-8")).hexdigest()

    async def _save_html(self, url, content):
        filename = self._hash_url(url) + ".html"
        filepath = os.path.join(self.html_dir, filename)
        try:
            async with aiofiles.open(
                filepath, "w", encoding="utf-8", errors="ignore"
            ) as f:
                await f.write(content)
            logging.debug(f"Saved HTML: {url} as {filename}")
        except OSError as e:
            logging.error(f"File save error {filepath} for URL {url}: {e}")
        except Exception as e:
            logging.error(f"Unexpected error saving file for URL {url}: {e}")

    async def _process_html(self, url, html_content):
        soup = BeautifulSoup(html_content, "lxml")
        found_links = 0
        for link in soup.find_all("a", href=True):
            href = link["href"]  # type: ignore
            normalized_url = self._normalize_url(url, href)

            if self._is_valid_url(normalized_url):
                self.graph.add_edge(url, normalized_url)

                if (
                    normalized_url not in self.visited_urls
                    and normalized_url not in self.in_progress_urls
                ):
                    if self.page_count + self.queue.qsize() < self.max_pages:
                        self.in_progress_urls.add(normalized_url)
                        await self.queue.put(normalized_url)
                        found_links += 1
                    else:
                        pass  # Max pages reached

        logging.debug(f"Found {found_links} new valid links on {url}")

    async def _worker(self):
        while True:
            async with self.semaphore:
                current_url = await self.queue.get()

                if current_url is None:
                    self.queue.task_done()
                    await self.queue.put(None)  # Signal other workers
                    break

                if current_url in self.visited_urls:
                    self.queue.task_done()
                    self.in_progress_urls.discard(current_url)
                    continue

                if self.page_count >= self.max_pages:
                    self.queue.task_done()
                    self.in_progress_urls.discard(current_url)
                    continue

                can_fetch = await self._can_fetch(current_url)
                if not can_fetch:
                    logging.info(f"Skipping (robots.txt): {current_url}")
                    self.visited_urls.add(current_url)
                    self.in_progress_urls.discard(current_url)
                    self.queue.task_done()
                    continue

                self.graph.add_node(current_url)

                logging.info(
                    f"Fetching ({self.page_count + 1}/{self.max_pages}): {current_url}"
                )
                try:
                    async with self.session.get(  # type: ignore
                        current_url, timeout=aiohttp.ClientTimeout(total=30)
                    ) as response:
                        self.visited_urls.add(current_url)
                        self.in_progress_urls.discard(current_url)
                        self.page_count += 1

                        if (
                            response.status == 200
                            and "text/html"
                            in response.headers.get("Content-Type", "").lower()
                        ):
                            html = await response.text(errors="ignore")
                            await self._save_html(current_url, html)
                            await self._process_html(current_url, html)
                        else:
                            logging.warning(
                                f"Fetch failed or non-HTML content for {current_url} (Status: {response.status}, Type: {response.headers.get('Content-Type')})"
                            )

                except aiohttp.ClientError as e:
                    logging.error(f"Network error fetching {current_url}: {e}")
                    self.visited_urls.add(current_url)
                    self.in_progress_urls.discard(current_url)
                except asyncio.TimeoutError:
                    logging.error(f"Timeout error fetching {current_url}")
                    self.visited_urls.add(current_url)
                    self.in_progress_urls.discard(current_url)
                except Exception as e:
                    logging.error(f"Unexpected error processing {current_url}: {e}")
                    self.visited_urls.add(current_url)
                    self.in_progress_urls.discard(current_url)
                finally:
                    self.queue.task_done()

    async def run(self):
        start_time = time.time()

        headers = {"User-Agent": self.user_agent}
        connector = aiohttp.TCPConnector(
            limit_per_host=max(1, int(self.semaphore._value / 2)),  # Ensure at least 1
            ssl=False,
        )
        async with aiohttp.ClientSession(
            headers=headers, connector=connector
        ) as session:
            self.session = session

            if self._is_valid_url(self.start_url):
                await self.queue.put(self.start_url)
                self.in_progress_urls.add(self.start_url)
                self.graph.add_node(self.start_url)
            else:
                logging.error(f"Start URL {self.start_url} is invalid or disallowed.")
                return

            workers = [
                asyncio.create_task(self._worker())
                for _ in range(self.semaphore._value)
            ]

            await self.queue.join()
            logging.info("Task queue empty.")

            await self.queue.put(None)  # Send termination signal

            await asyncio.gather(*workers, return_exceptions=True)
            logging.info("All workers finished.")

        try:
            nx.write_graphml(self.graph, self.graph_file)
            logging.info(f"Link graph saved to: {self.graph_file}")
        except Exception as e:
            logging.error(f"Failed to save graph to {self.graph_file}: {e}")

        end_time = time.time()
        duration = end_time - start_time
        logging.info("Crawler finished.")
        logging.info(f"Fetched {self.page_count} pages.")
        logging.info(
            f"Visited (incl. errors/robots): {len(self.visited_urls)} unique URLs."
        )
        logging.info(
            f"Graph contains {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges."
        )
        logging.info(f"Total execution time: {duration:.2f} seconds.")


def main():
    parser = argparse.ArgumentParser(description="Asynchronous Web Crawler")
    parser.add_argument("start_url", help="Initial URL (e.g., https://www.osu.edu)")
    parser.add_argument(
        "-m", "--max_pages", type=int, default=3000, help="Max pages to fetch"
    )
    parser.add_argument(
        "-c", "--concurrency", type=int, default=10, help="Concurrent download tasks"
    )
    parser.add_argument(
        "-o", "--output_dir", default="./artifacts", help="Output directory for results"
    )
    parser.add_argument(
        "-ua", "--user_agent", default=DEFAULT_USER_AGENT, help="HTTP User-Agent"
    )

    args = parser.parse_args()

    parsed_url = urlparse(args.start_url)
    if not parsed_url.scheme or not parsed_url.netloc:
        print(f"Error: Invalid start URL format: {args.start_url}")
        print("Expected format like 'http://domain.com' or 'https://domain.com'")
        return

    if args.concurrency <= 0:
        print("Error: Concurrency must be a positive integer.")
        return
    if args.max_pages <= 0:
        print("Error: Max pages must be a positive integer.")
        return

    crawler = AsyncCrawler(
        start_url=args.start_url,
        max_pages=args.max_pages,
        concurrency=args.concurrency,
        output_dir=args.output_dir,
        user_agent=args.user_agent,
    )

    try:
        asyncio.run(crawler.run())
    except KeyboardInterrupt:
        logging.info("Interrupted by user (Ctrl+C).")
    except Exception as e:
        logging.exception(f"An unexpected top-level error occurred: {e}")


if __name__ == "__main__":
    # Optional: Set event loop policy for Windows if needed
    # if os.name == 'nt':
    #     asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    main()
