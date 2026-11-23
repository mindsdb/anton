---
name: public-data-sources
description: 'Recall BEFORE fetching public news, market, economic, or world data in scratchpad: free no-API-key RSS, financial (yfinance, FRED, CoinGecko), economic (World Bank, OECD, FX), and social endpoints.'
metadata:
  display_name: Public data & world-event sources
  provenance: builtin
---
PUBLIC DATA AND WORLD EVENTS (use these by default — no API keys required):
Start with free, open sources. Only ask the user to connect paid services or personal accounts if they request it or if free sources are insufficient.

News & current events (via RSS — parse with BeautifulSoup and lxml):
- Google News RSS: `https://news.google.com/rss/search?q={query}&hl={lang}&gl={country}` — any topic, any country. Use country/language codes (gl=US&hl=en, gl=MX&hl=es, gl=BR&hl=pt-BR, gl=JP&hl=ja, etc.). This is your primary news source.
- Reuters: `https://www.rss.reuters.com/news/` (world, business, tech sections)
- AP News: `https://rsshub.app/apnews/topics/{topic}` (top-news, politics, business, technology, science, entertainment)
- BBC World: `http://feeds.bbci.co.uk/news/rss.xml` (also /world, /business, /technology)
- NPR: `https://feeds.npr.org/1001/rss.xml` (news), `1006/rss.xml` (business)
- For country-specific news, use Google News RSS with the country code — it aggregates local sources automatically.
- Parse a feed with `BeautifulSoup(response.content, "xml")` (needs `lxml`): each `item` has `title`, `link`, `pubDate` and `description`. Store them as a list of dicts for dashboard integration. Don't use feedparser; it is usually not installed.

Financial & market data:
- yfinance: stocks, ETFs, indices, crypto, forex — historical and real-time. Use tickers like ^GSPC (S&P 500), ^DJI (Dow), ^IXIC (Nasdaq), BTC-USD, etc. It is usually not installed: list `yfinance` in the cell's `packages` array, and it is installed before the cell runs.
- FRED (Federal Reserve): `https://fred.stlouisfed.org/` — macro indicators (GDP, CPI, unemployment, interest rates, money supply). Fetch the CSV directly: `https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}` (no key needed).
- CoinGecko: `https://api.coingecko.com/api/v3/` — crypto prices, market cap, volume, trending coins. Free, no key.

Economic & global data:
- World Bank: `https://api.worldbank.org/v2/country/{code}/indicator/{indicator}?format=json` — GDP, population, poverty, education, health by country. Free, no key.
- OECD: `https://sdmx.oecd.org/public/rest/data/` — economic indicators for OECD countries.
- Open Exchange Rates: `https://open.er-api.com/v6/latest/{base}` — free forex rates.

Social & sentiment:
- Reddit JSON: `https://www.reddit.com/r/{subreddit}/.json` — add .json to any Reddit URL for structured data. Good for sentiment on specific topics.
- HackerNews: `https://hacker-news.firebaseio.com/v0/` — tech news, top/new/best stories.

When building "state of affairs" or country dashboards, ALWAYS layer multiple sources: quantitative data (markets, economic indicators) + news context (RSS headlines) + narrative synthesis. A chart without news context is just numbers; headlines without data are just opinions. Combine them.
