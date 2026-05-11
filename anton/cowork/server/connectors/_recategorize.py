"""One-shot category fixup.

Why: the LLM-driven generator picked categories from the schema's
allowed-list in the prompt — but we kept adding new categories
(observability, ai, web-search, maps, public-data, mobility,
logistics, hr, accounting, database, vector-db, engineering, cloud)
without updating the schema. Result: ~85 connectors landed in `other`
or `data-warehouse` / `data` / `analytics` instead of their proper
home.

This script remaps every connector to the right category based on
its `id`, so the picker shows them under the right section.
"""
from __future__ import annotations
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent

CATEGORY_MAP = {
    # CRM
    "pipedrive": "crm", "close": "crm", "copper": "crm", "attio": "crm", "folk": "crm",
    "hubspot": "crm", "salesforce": "crm",
    # Sales engagement
    "outreach": "sales-engagement", "salesloft": "sales-engagement",
    "apollo": "sales-engagement", "lemlist": "sales-engagement",
    "reply_io": "sales-engagement", "smartlead": "sales-engagement",
    "instantly": "sales-engagement",
    # Lead enrichment
    "clay": "enrichment", "clearbit": "enrichment", "zoominfo": "enrichment",
    "lusha": "enrichment",
    # Marketing automation
    "mailchimp": "marketing", "customer_io": "marketing", "marketo": "marketing",
    "iterable": "marketing", "braze": "marketing", "klaviyo": "marketing",
    # Product & web analytics
    "amplitude": "analytics", "mixpanel": "analytics", "heap": "analytics",
    "posthog": "analytics", "google_analytics_4": "analytics",
    "google_search_console": "analytics", "plausible": "analytics", "fathom": "analytics",
    # Ads
    "google_ads": "ads", "linkedin_ads": "ads", "meta_ads": "ads",
    # Support / helpdesk
    "intercom": "support", "zendesk": "support", "freshdesk": "support",
    "helpscout": "support", "front": "support",
    # Customer success
    "gainsight": "customer-success", "vitally": "customer-success",
    "churnzero": "customer-success",
    # Revenue intel
    "gong": "revenue-intel", "chorus_ai": "revenue-intel", "clari": "revenue-intel",
    # Communication
    "gmail": "communication", "outlook": "communication", "slack": "communication",
    "microsoft_teams": "communication", "discord": "communication",
    # Productivity & project mgmt
    "linear": "productivity", "asana": "productivity", "jira": "productivity",
    "confluence": "productivity", "trello": "productivity", "clickup": "productivity",
    "notion": "productivity", "monday": "productivity", "airtable": "productivity",
    "coda": "productivity", "todoist": "productivity", "basecamp": "productivity",
    "google_calendar": "productivity",
    # Scheduling
    "calendly": "scheduling", "chili_piper": "scheduling",
    # Forms
    "typeform": "forms",
    # Documents
    "docusign": "documents", "pandadoc": "documents",
    # Billing
    "stripe": "billing", "chargebee": "billing", "recurly": "billing",
    # Accounting
    "quickbooks": "accounting", "xero": "accounting", "sage_intacct": "accounting",
    "freshbooks": "accounting", "netsuite": "accounting", "zoho_books": "accounting",
    # HR
    "fifteen_five": "hr", "lattice": "hr", "bamboohr": "hr", "gusto": "hr",
    "rippling_hris": "hr", "deel": "hr", "hibob": "hr", "personio": "hr",
    "greenhouse": "hr", "lever": "hr", "ashby": "hr", "workable": "hr",
    # Files
    "google_drive": "files",
    # Mobility
    "uber": "mobility", "lyft": "mobility", "doordash": "mobility",
    "instacart": "mobility", "bolt": "mobility", "grab": "mobility",
    # Logistics
    "shipstation": "logistics", "shippo": "logistics", "easypost": "logistics",
    "shipbob": "logistics",
    # AI APIs
    "openai": "ai", "anthropic": "ai", "google_gemini": "ai", "cohere": "ai",
    "mistral": "ai", "huggingface": "ai", "replicate": "ai", "together_ai": "ai",
    "groq": "ai", "fireworks": "ai", "elevenlabs": "ai", "deepgram": "ai",
    "assemblyai": "ai", "stability_ai": "ai", "runway": "ai", "pika": "ai", "luma": "ai",
    # Web search
    "exa": "web-search", "tavily": "web-search", "google_search": "web-search",
    "bing_search": "web-search", "brave_search": "web-search", "serper": "web-search",
    "serpapi": "web-search", "perplexity": "web-search", "you_com": "web-search",
    "kagi": "web-search",
    # Maps
    "google_maps": "maps", "mapbox": "maps", "here_maps": "maps", "tomtom": "maps",
    "openstreetmap": "maps",
    # Public data
    "newsapi": "public-data", "openweather": "public-data",
    "tomorrow_io": "public-data", "accuweather": "public-data",
    "alphavantage": "public-data", "polygon_io": "public-data",
    "coingecko": "public-data", "etherscan": "public-data",
    "youtube_data": "public-data", "spotify": "public-data",
    # Engineering / DevOps
    "github": "engineering", "gitlab": "engineering", "bitbucket": "engineering",
    "vercel": "engineering", "netlify": "engineering", "circleci": "engineering",
    "launchdarkly": "engineering", "figma": "engineering",
    # Observability
    "datadog": "observability", "newrelic": "observability",
    "sentry": "observability", "pagerduty": "observability",
    "opsgenie": "observability", "grafana": "observability",
    "prometheus": "observability", "honeycomb": "observability",
    "splunk": "observability", "dynatrace": "observability",
    "sumo_logic": "observability", "better_stack": "observability",
    "statuspage": "observability", "pingdom": "observability",
    "uptimerobot": "observability", "bugsnag": "observability",
    "rollbar": "observability", "otel_collector": "observability",
    "jaeger": "observability", "zipkin": "observability",
    "loki": "observability", "tempo": "observability",
    "victoriametrics": "observability", "aws_cloudwatch": "observability",
    "gcp_cloud_monitoring": "observability", "azure_monitor": "observability",
    "elastic_apm": "observability", "graylog": "observability",
    "logz_io": "observability", "coralogix": "observability",
    "checkly": "observability", "appdynamics": "observability",
    "instana": "observability",
    # Databases (operational + warehouse + variants)
    "postgres": "database", "mysql": "database", "mariadb": "database",
    "mongodb": "database", "mssql": "database", "oracle": "database",
    "redis": "database", "elasticsearch": "database", "cassandra": "database",
    "clickhouse": "database", "cockroachdb": "database", "neo4j": "database",
    "couchbase": "database", "dynamodb": "database", "firestore": "database",
    "influxdb": "database", "snowflake": "database", "bigquery": "database",
    "redshift": "database", "databricks": "database",
    "neon": "database", "supabase": "database",
    "aws_rds_postgres": "database", "aws_aurora_postgres": "database",
    "gcp_cloudsql_postgres": "database", "azure_postgres": "database",
    "heroku_postgres": "database", "planetscale": "database",
    "aws_rds_mysql": "database", "aws_aurora_mysql": "database",
    "gcp_cloudsql_mysql": "database", "azure_mysql": "database",
    "azure_sql": "database", "aws_rds_sqlserver": "database",
    "mongodb_atlas": "database", "azure_cosmos_db": "database",
    "gcp_spanner": "database", "upstash": "database",
    # Vector DBs
    "pinecone": "vector-db", "weaviate": "vector-db",
    "qdrant": "vector-db", "chroma": "vector-db",
    # Data infra (CDP / reverse-ETL)
    "segment": "data", "hightouch": "data", "census": "data", "rudderstack": "data",
    # Cloud providers
    "aws": "cloud", "gcp": "cloud", "azure": "cloud", "cloudflare": "cloud",
    "digitalocean": "cloud", "heroku": "cloud", "render": "cloud", "fly_io": "cloud",
}


def main():
    fixed, unchanged, missing = 0, 0, []
    for path in sorted(OUT.glob("*.json")):
        cid = path.stem
        if cid not in CATEGORY_MAP:
            missing.append(cid)
            continue
        spec = json.loads(path.read_text(encoding="utf-8"))
        target = CATEGORY_MAP[cid]
        # Top-level category
        cat_changed = spec.get("category") != target
        if cat_changed:
            spec["category"] = target
        # Form's logo_color is unrelated; leave it.
        # Persist if anything changed.
        if cat_changed:
            path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
            fixed += 1
        else:
            unchanged += 1
    print(f"fixed: {fixed}, already correct: {unchanged}, no mapping: {len(missing)}")
    if missing:
        print("Connectors with no mapping (left as-is):")
        for c in missing:
            print(f"  - {c}")


if __name__ == "__main__":
    main()
