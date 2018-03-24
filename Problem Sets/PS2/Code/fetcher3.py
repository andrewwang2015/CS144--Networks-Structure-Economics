#!/usr/bin/env python
"""
A simple program to extract links in a web page, by Minghong Lin, for CMS/CS/EE 144.
Remade for Python 3 by: Catherine Ma (cmma@caltech.edu)

If you use Python, you can import this file and simply call the function
"fetch_links(URL)" to get a (may be empty) list, which contains all hyperlinks
in the web page specified by the URL. It returns None if the URL is not a valid
HTML page or some error occurred.
If you use another programming language, you can make a system call to execute this
file and pipe the output to your program. The command is
"python fetcher.py http://..."
The output is a list (may be empty "[]") of hyperlinks which looks like:
"['http://today.caltech.edu', 'http://www.caltech.edu/copyright/', ...]"
The output is "None" if the URL is not a valid HTML page or some error occurred.
"""

from html.parser import HTMLParser
from urllib.parse import urljoin
from urllib import request
from urllib.error import URLError
import urllib

urls = []

# Our version of the HTMLParser, which handles start tags differently than Python's
# own HTMLParser. We overwrite the handle_starttag method to look for the desired
# hyperlinks.
class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        if tag == 'a' or tag == 'area': # extract href links from "a" and "area" tags
            href = [v for (k, v) in attrs if k == 'href']
            urls.extend(href)
        if tag == 'frame' or tag == 'iframe': # extra src links from "frame" and "iframe" tags
            src = [v for (k, v) in attrs if k == 'src']
            urls.extend(src)

    # Refine the hyperlinks: ignore the parameters in dynamic URLs, convert
    # all hyperlinks to absolute URLs, and remove duplicated URLs.
    def get_links(self, current_url):
        res = set()       # use a set to avoid duplicated URLs
        for url in urls:
            url = url.split('?')[0]   # extract the part before '?' if any
            url = url.split('#')[0]   # extract the part before '#' if any
            if url.startswith("http://") or url.startswith("https://"):
                res.add(url)
            elif url.startswith("\\"):
                url = url.strip()[2:-2]
                if url.startswith("http://") or url.startswith("https://"):
                    res.add(url)
                elif 'javascript:void(0)' not in url:
                    res.add(urljoin(current_url, url))
            else:
                res.add(urljoin(current_url, url))
        res.discard(current_url)    # self-link is removed
        return list(res)

# Fetch an HTML file and return the real (redirected) URL and the content.
def fetch_html_page(url):
    content = None
    real_url = url
    try:
        usock = urllib.request.urlopen(url, timeout=10)
        real_url = usock.url   # real_url will be changed if there is a redirection 
        if "text/html" in usock.info()['content-type']:
            content=usock.read()    # only fetch it if it is html (not mp3/avi/...)
        usock.close()
    # Terminate on CTRL+C sequences, and pass URLError up the stack.
    except (KeyboardInterrupt, urllib.error.URLError):
        raise
    except urllib.error.HTTPError:  
        pass
    return (real_url, content)

# Fetch the hyperlinks by first fetching the content then using our HTMLParser to
# parse them.
def fetch_links(url):
    global urls
    urls = []
    links = None
    try:
        parser = MyHTMLParser()
        real_url, content = fetch_html_page(url)
        if content is not None:
            parser.urls = []
            parser.feed(str(content))
            parser.close()
            links = parser.get_links(real_url)
    # Terminate on CTRL+C sequences, and pass URLError up the stack.
    except (KeyboardInterrupt, URLError):
        raise
    except:
        pass
    return links

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2 or not sys.argv[1].startswith("http://"):
        prog = sys.argv[0]
        print("usage: %s http://...\tShows all hyperlinks in an HTML page." % prog)
    else:
        print(fetch_links(sys.argv[1]))
