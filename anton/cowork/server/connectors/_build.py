"""LLM-driven connector spec generator.

Why LLM-driven: hand-written specs go stale (vendor portals reorganize,
scopes change, deprecations land). Letting Claude write each spec
keeps the docs honest with current vendor reality, and lets us add
~50 connectors in one batch without writing each by hand.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...    # or ANTON_ANTHROPIC_API_KEY
    python3 server/connectors/_build.py            # generate missing only
    python3 server/connectors/_build.py --force    # overwrite everything
    python3 server/connectors/_build.py --only slack,stripe   # subset

Each TARGET below is a tuple of (id, label, hint, suggested_logo).
The LLM uses gmail.json as a few-shot example and produces a full
DataVaultForm spec following the same shape.

Protected files (gmail, google_drive, google_calendar, hubspot,
posthog, salesforce) are NEVER touched — those were hand-iterated
with the user and should stay editable.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import httpx


OUT_DIR = Path(__file__).resolve().parent
PROTECTED = {"gmail", "google_drive", "google_calendar", "hubspot", "posthog", "salesforce"}

# Defaults — overridable via env. Sonnet is plenty for structured JSON.
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-5"
MAX_TOKENS = 8192

# Available icons in the renderer's Ico palette. The LLM MUST pick
# from this list — anything else falls back to `database`.
LOGOS = [
    "search", "chats", "list", "grid", "image", "sidebar", "menu",
    "sun", "moon", "power", "copy", "refresh", "code", "plus",
    "folder", "phone", "clock", "sparkle", "slider", "settings",
    "pin", "mic", "send", "stop", "attach", "download", "check",
    "more", "edit", "trash", "schedule", "doc", "globe", "brain",
    "database", "mail", "upload", "wifi", "key", "mindsdb", "link",
    "cube",
]


# ─── TARGETS ──────────────────────────────────────────────────────────
# (id, label, one-line hint, suggested logo)
# The LLM fills in everything else — aliases, keywords, methods,
# fields, how_to, oauth blocks, help_url. Hints are the minimum the
# LLM needs to disambiguate (e.g. "paste API key, no OAuth needed").

TARGETS: list[tuple[str, str, str, str]] = [
    # ── CRM ──
    ("pipedrive", "Pipedrive",
     "Sales-focused CRM. Auth: paste personal API token + company subdomain.",
     "link"),
    ("close", "Close",
     "Sales-focused CRM with calling/email built in. Auth: paste API key from Settings → API Keys.",
     "link"),
    ("copper", "Copper",
     "CRM tightly integrated with Google Workspace. Auth: paste API key + login email (both required on every request).",
     "link"),
    ("attio", "Attio",
     "Modern flexible CRM with custom objects. Auth: paste Access Token from Settings → Apps & Integrations → Developer.",
     "link"),
    ("folk", "Folk",
     "Relationship-focused CRM. Auth: paste API key from Settings → Developers.",
     "link"),

    # ── Sales engagement ──
    ("outreach", "Outreach",
     "Sales engagement / sequences platform. NO API key path — OAuth 2.0 only (BYOK Outreach app). PKCE-compatible.",
     "send"),
    ("salesloft", "Salesloft",
     "Sales engagement / cadences platform. NO API key path — OAuth 2.0 only (BYOK Salesloft app).",
     "send"),
    ("apollo", "Apollo.io",
     "Outbound prospecting + sequences. Auth: paste API key from Settings → Integrations → API.",
     "send"),
    ("lemlist", "Lemlist",
     "Cold email / outbound platform. Auth: paste API key from team settings.",
     "send"),
    ("reply_io", "Reply.io",
     "Multi-channel sales engagement. Auth: paste API key from Settings → API & Integrations.",
     "send"),
    ("smartlead", "Smartlead",
     "Cold email + deliverability platform. Auth: paste API key from Settings → API.",
     "send"),
    ("instantly", "Instantly",
     "Cold email platform. Auth: paste API key from Integrations → API.",
     "send"),

    # ── Lead enrichment ──
    ("clay", "Clay",
     "Data enrichment + waterfalls platform for outbound. Auth: paste API key from Workspace settings → API keys.",
     "search"),
    ("clearbit", "Clearbit",
     "Person/company enrichment (now HubSpot Breeze Intelligence). Auth: paste secret API key (sk_…) from dashboard.",
     "search"),
    ("zoominfo", "ZoomInfo",
     "B2B contact + intent data. Auth: PKI client credentials (username + password/key) issued by your account admin — NOT self-serve.",
     "search"),
    ("lusha", "Lusha",
     "Contact enrichment for emails + phone numbers. Auth: paste API key from Profile → API Access.",
     "search"),

    # ── Marketing automation / email ──
    ("mailchimp", "Mailchimp",
     "Email marketing / audiences. Auth: paste API key (ends in -usXX datacenter suffix).",
     "mail"),
    ("customer_io", "Customer.io",
     "Lifecycle messaging. Auth: paste Track site ID + Track API key, optionally App API key. Region selector (us|eu).",
     "mail"),
    ("marketo", "Marketo",
     "Adobe Marketo Engage. Auth: Custom Service in LaunchPoint — paste client ID, client secret, REST endpoint URL.",
     "mail"),
    ("iterable", "Iterable",
     "Cross-channel messaging. Auth: paste server-side API key from Integrations → API Keys.",
     "mail"),
    ("braze", "Braze",
     "Customer engagement / messaging platform. Auth: paste REST API key + REST endpoint URL (per-instance).",
     "mail"),
    ("klaviyo", "Klaviyo",
     "Ecommerce email + SMS marketing. Auth: paste Private API key (pk_…). Site ID is for client-side ingestion only — don't use that.",
     "mail"),

    # ── Product analytics ──
    ("amplitude", "Amplitude",
     "Product analytics — events, funnels, retention. Auth: paste project API key + secret key. Region selector (us|eu).",
     "sparkle"),
    ("mixpanel", "Mixpanel",
     "Product analytics. Auth: paste service-account username + secret + project ID. Region selector (us|eu).",
     "sparkle"),
    ("heap", "Heap",
     "Product analytics with autocapture. Auth: paste app ID + server-side API key.",
     "sparkle"),
    ("google_analytics_4", "Google Analytics 4",
     "Web/app analytics from Google. Auth: OAuth 2.0 (BYOK Google Cloud client). Same Google OAuth block as Drive/Calendar — auth_url=https://accounts.google.com/o/oauth2/v2/auth, token_url=https://oauth2.googleapis.com/token, extra_auth_params={access_type:offline, prompt:consent}. Scopes: analytics.readonly + userinfo.email.",
     "sparkle"),
    ("google_search_console", "Google Search Console",
     "SEO performance data from Google. Auth: OAuth 2.0 (BYOK Google Cloud client). Same Google OAuth block as Drive. Scopes: webmasters.readonly + userinfo.email.",
     "search"),
    ("plausible", "Plausible",
     "Privacy-friendly web analytics. Auth: paste API key from User Settings → API Keys + site ID (the domain).",
     "sparkle"),
    ("fathom", "Fathom",
     "Privacy-friendly web analytics. Auth: paste API key + 8-char site ID (from the dashboard URL).",
     "sparkle"),

    # ── Ads ──
    ("google_ads", "Google Ads",
     "Search/display ad campaigns. Auth: OAuth 2.0 (BYOK Google Cloud client) PLUS a developer token from Google Ads MCC. Both required. Scopes: adwords + userinfo.email. Optional login_customer_id for MCC manager accounts.",
     "sparkle"),
    ("linkedin_ads", "LinkedIn Ads",
     "B2B ads (Campaign Manager). Auth: OAuth 2.0, BYOK app from LinkedIn Developer Portal — requires Marketing Developer Platform approval. auth_url=https://www.linkedin.com/oauth/v2/authorization, token_url=https://www.linkedin.com/oauth/v2/accessToken. Scopes: r_ads, r_ads_reporting, rw_ads.",
     "sparkle"),
    ("meta_ads", "Meta Ads",
     "Facebook + Instagram ads. Auth: OAuth 2.0, BYOK app from Meta for Developers (Marketing API). auth_url=https://www.facebook.com/v19.0/dialog/oauth, token_url=https://graph.facebook.com/v19.0/oauth/access_token. Scopes: ads_read, ads_management, business_management.",
     "sparkle"),

    # ── Customer support ──
    ("intercom", "Intercom",
     "Messaging / customer support / help center. Auth: paste Access Token from Developer Hub → app → Authentication. Region selector (us|eu|au).",
     "chats"),
    ("zendesk", "Zendesk",
     "Tickets / help desk. Auth: paste API token + login email + subdomain. Token from Admin Center → Apps and integrations → APIs → Zendesk API.",
     "chats"),
    ("freshdesk", "Freshdesk",
     "Tickets / help desk. Auth: paste API key + subdomain (uses HTTP basic with key as username).",
     "chats"),
    ("helpscout", "Help Scout",
     "Shared inbox / help desk. Auth: OAuth2 client credentials — paste App ID + App Secret (no browser flow needed for client_credentials grant).",
     "chats"),
    ("front", "Front",
     "Shared inbox / customer comms. Auth: paste API token from Settings → Developers → API tokens.",
     "chats"),

    # ── Customer success ──
    ("gainsight", "Gainsight",
     "Customer success / health scoring. Auth: paste Access Key from Administration → Connectors 2.0 → Auth → Access Keys, plus your tenant URL.",
     "check"),
    ("vitally", "Vitally",
     "Customer success / playbooks. Auth: paste API key + subdomain.",
     "check"),
    ("churnzero", "ChurnZero",
     "Customer success / engagement scoring. Auth: paste two paired keys — API key + AppKey (different panels in admin).",
     "check"),

    # ── Revenue intelligence ──
    ("gong", "Gong",
     "Call recording + revenue intel. Auth: paste Access Key + Access Key Secret from Company Settings → Ecosystem → API. Secret shown once.",
     "mic"),
    ("chorus_ai", "Chorus.ai",
     "Call recording + revenue intel (now ZoomInfo Chorus). Auth: paste API key from Settings → Integrations → API.",
     "mic"),
    ("clari", "Clari",
     "Forecasting + pipeline intel. Auth: paste API key from Settings → Integrations → API + tenant slug. Often admin/CSM-gated.",
     "sparkle"),

    # ── Communication ──
    ("slack", "Slack",
     "Workspace messaging. Auth: paste Bot User OAuth Token (xoxb-…) from a Slack app installed in your workspace. Recommend bot-token over OAuth flow for desktop simplicity.",
     "chats"),
    ("microsoft_teams", "Microsoft Teams",
     "Microsoft messaging / collab. Auth: OAuth 2.0 against Microsoft Identity (BYOK app from Azure AD / Entra ID). auth_url=https://login.microsoftonline.com/common/oauth2/v2.0/authorize, token_url=https://login.microsoftonline.com/common/oauth2/v2.0/token. Scopes: Chat.ReadWrite, ChannelMessage.Send, Team.ReadBasic.All, User.Read, offline_access.",
     "chats"),
    ("discord", "Discord",
     "Community / server messaging. Auth: paste Bot Token from Discord Developer Portal → app → Bot. Bot must be invited to server via OAuth2 URL Generator.",
     "chats"),

    # ── Data warehouse ──
    ("snowflake", "Snowflake",
     "Cloud data warehouse. Auth: paste user + password + account identifier (looks like abc12345.us-east-1). Optional default warehouse / database / schema / role. Note: key-pair auth not yet supported — request if needed.",
     "database"),
    ("bigquery", "BigQuery",
     "Google Cloud data warehouse. Auth: paste service-account JSON key (textarea, secret) + project ID. Same shape as Drive/Calendar service-account method.",
     "database"),
    ("redshift", "Redshift",
     "AWS data warehouse. Auth: paste host (cluster endpoint), port (default 5439), database, user, password. IAM auth not yet supported.",
     "database"),
    ("databricks", "Databricks",
     "Lakehouse / SQL warehouses. Auth: paste workspace URL + personal access token (dapi…) + SQL warehouse HTTP path.",
     "database"),

    # ── Reverse ETL / CDP ──
    ("segment", "Segment",
     "Customer data platform / event ingestion. Auth: paste source Write Key (required) + workspace access token (optional, for Public API).",
     "database"),
    ("hightouch", "Hightouch",
     "Reverse ETL / data activation. Auth: paste API token from Workspace settings → API tokens.",
     "database"),
    ("census", "Census",
     "Reverse ETL. Auth: paste API key from Settings → API.",
     "database"),
    ("rudderstack", "RudderStack",
     "Open-source CDP / event ingestion. Auth: paste source Write Key + Data Plane URL.",
     "database"),

    # ── Scheduling ──
    ("calendly", "Calendly",
     "Meeting scheduling. Auth: paste Personal Access Token from Integrations → API & Webhooks.",
     "schedule"),
    ("chili_piper", "Chili Piper",
     "Inbound meeting routing + booking. Auth: paste API key from Admin → API.",
     "schedule"),

    # ── Forms ──
    ("typeform", "Typeform",
     "Forms + surveys. Auth: paste personal access token (tfp_…) from Settings → Personal tokens.",
     "doc"),

    # ── Documents / contracts ──
    ("docusign", "DocuSign",
     "E-signature. Auth: OAuth 2.0 BYOK (Integration Key from DocuSign Admin → Apps and Keys). auth_url=https://account.docusign.com/oauth/auth, token_url=https://account.docusign.com/oauth/token. Scopes: signature, extended. Note demo vs production hostnames differ.",
     "doc"),
    ("pandadoc", "PandaDoc",
     "Document workflow + e-signature. Auth: paste API key from Settings → API and integrations. Simpler than OAuth flow, recommend that.",
     "doc"),

    # ── Billing ──
    ("stripe", "Stripe",
     "Payments + subscriptions. Auth: paste a RESTRICTED key (rk_live_… or rk_test_…) — NOT a secret key. Restricted keys can be scoped to specific resources/permissions.",
     "key"),
    ("chargebee", "Chargebee",
     "Subscription billing. Auth: paste API key (live_…) + site name (subdomain).",
     "key"),
    ("recurly", "Recurly",
     "Subscription billing. Auth: paste private API key from Integrations → API Credentials.",
     "key"),
    ("quickbooks", "QuickBooks Online",
     "Accounting. Auth: OAuth 2.0 BYOK (app from Intuit Developer). auth_url=https://appcenter.intuit.com/connect/oauth2, token_url=https://oauth.platform.intuit.com/oauth2/v1/tokens/bearer. Scope: com.intuit.quickbooks.accounting.",
     "key"),

    # ── Communication (additions) ──
    ("outlook", "Outlook",
     "Microsoft 365 mail. Auth: OAuth 2.0 via Microsoft Graph (BYOK Azure AD app). auth_url=https://login.microsoftonline.com/common/oauth2/v2.0/authorize, token_url=https://login.microsoftonline.com/common/oauth2/v2.0/token. Scopes: Mail.ReadWrite, Mail.Send, User.Read, offline_access.",
     "mail"),

    # ── Productivity / Project Management ──
    ("linear", "Linear",
     "Issue tracking + product engineering. Auth: paste personal API key from Settings → API → Personal API keys.",
     "list"),
    ("asana", "Asana",
     "Work management / projects + tasks. Auth: paste Personal Access Token (PAT) from My Settings → Apps → Manage Developer Apps.",
     "list"),
    ("jira", "Jira",
     "Issue tracking from Atlassian. Auth: paste API token + login email + Atlassian subdomain (yourorg.atlassian.net). HTTP basic auth uses email as username.",
     "list"),
    ("confluence", "Confluence",
     "Documentation/wiki from Atlassian. Same auth shape as Jira: API token + login email + Atlassian subdomain.",
     "doc"),
    ("trello", "Trello",
     "Kanban-style project boards from Atlassian. Auth: paste API key + token (both from trello.com/app-key).",
     "grid"),
    ("clickup", "ClickUp",
     "Project + task management. Auth: paste personal API token from Settings → Apps.",
     "list"),
    ("notion", "Notion",
     "Docs + databases. Auth: paste Internal Integration Secret from Notion → Settings & Members → Connections → Develop or manage integrations.",
     "doc"),
    ("monday", "Monday.com",
     "Work OS / project management. Auth: paste API v2 token from Profile → Admin → API.",
     "grid"),
    ("airtable", "Airtable",
     "Spreadsheet/database hybrid. Auth: paste Personal Access Token (PAT) from account → Developer hub → Personal access tokens (the older API keys are deprecated).",
     "grid"),
    ("coda", "Coda",
     "Docs + tables + automations. Auth: paste API token from Account settings → API tokens.",
     "doc"),
    ("todoist", "Todoist",
     "Personal + team task management. Auth: paste API token from Settings → Integrations → Developer → API token.",
     "check"),
    ("basecamp", "Basecamp",
     "Project management + team comms. Auth: OAuth 2.0 BYOK (app from launchpad.37signals.com/integrations). auth_url=https://launchpad.37signals.com/authorization/new, token_url=https://launchpad.37signals.com/authorization/token.",
     "folder"),

    # ── Engineering / DevOps ──
    ("github", "GitHub",
     "Source code + PRs + issues. Auth: paste fine-grained personal access token (github_pat_…) from Settings → Developer settings → Personal access tokens. Recommend fine-grained over classic. OAuth is also supported but PAT is simpler.",
     "code"),
    ("gitlab", "GitLab",
     "Source code + CI + issues. Auth: paste personal access token (glpat-…) from User Settings → Access Tokens. Optional self-hosted instance URL.",
     "code"),
    ("bitbucket", "Bitbucket",
     "Atlassian source code + PRs. Auth: paste username + app password (NOT account password). App password from Settings → App passwords.",
     "code"),
    ("sentry", "Sentry",
     "Error monitoring. Auth: paste auth token (sntrys_…) from Settings → Auth Tokens. Optional self-hosted URL.",
     "brain"),
    ("datadog", "Datadog",
     "Observability / metrics / logs. Auth: paste API key + Application key. Region selector (us1|us3|us5|eu|us1-fed).",
     "sparkle"),
    ("newrelic", "New Relic",
     "Observability / APM. Auth: paste User API key (NRAK-…) from Profile → API keys. Region selector (us|eu).",
     "sparkle"),
    ("pagerduty", "PagerDuty",
     "Incident management + on-call. Auth: paste REST API key from Integrations → API Access Keys.",
     "phone"),
    ("opsgenie", "Opsgenie",
     "Atlassian incident management + on-call. Auth: paste API key from Settings → Integration list → API → Add new integration.",
     "phone"),
    ("vercel", "Vercel",
     "Frontend hosting + deployments. Auth: paste personal API token from Settings → Tokens.",
     "globe"),
    ("netlify", "Netlify",
     "Frontend hosting + deployments. Auth: paste personal access token from User settings → Applications → Personal access tokens.",
     "globe"),
    ("circleci", "CircleCI",
     "CI/CD pipelines. Auth: paste personal API token from User Settings → Personal API Tokens.",
     "refresh"),
    ("launchdarkly", "LaunchDarkly",
     "Feature flags + experimentation. Auth: paste API access token (api-…) from Account settings → Authorization → Access tokens.",
     "slider"),
    ("figma", "Figma",
     "Design files + prototypes. Auth: paste personal access token from Settings → Personal access tokens.",
     "image"),

    # ── Databases — operational ──
    ("postgres", "PostgreSQL",
     "Open-source relational database. Auth: paste host + port (default 5432) + database + user + password. Supports SSL toggle.",
     "database"),
    ("mysql", "MySQL",
     "Open-source relational database. Auth: paste host + port (default 3306) + database + user + password. Supports SSL toggle.",
     "database"),
    ("mariadb", "MariaDB",
     "MySQL-compatible relational database. Auth: paste host + port (default 3306) + database + user + password. Supports SSL toggle.",
     "database"),
    ("mongodb", "MongoDB",
     "Document database. Auth: paste connection string (mongodb:// or mongodb+srv://) — single field is simplest. Optionally fall back to host+user+password+database fields.",
     "database"),
    ("mssql", "SQL Server",
     "Microsoft SQL Server / Azure SQL. Auth: paste host + port (default 1433) + database + user + password. Supports SSL/encrypt toggle.",
     "database"),
    ("oracle", "Oracle Database",
     "Enterprise relational database. Auth: paste host + port (default 1521) + service name (or SID) + user + password.",
     "database"),
    ("redis", "Redis",
     "In-memory key-value store. Auth: paste connection string (redis:// or rediss://) — single field. Or fall back to host + port + password + database number.",
     "database"),
    ("elasticsearch", "Elasticsearch",
     "Search engine + log database. Auth: paste cluster URL + either (username + password) or API key. Mark API key as recommended (simpler than user/password + role setup).",
     "search"),
    ("cassandra", "Cassandra",
     "Distributed wide-column database. Auth: paste contact points (comma-separated host:port) + keyspace + username + password.",
     "database"),
    ("clickhouse", "ClickHouse",
     "Columnar OLAP database. Auth: paste host + port (default 8443 HTTPS) + database + user + password. Supports HTTPS toggle. Optional self-hosted vs ClickHouse Cloud.",
     "database"),
    ("cockroachdb", "CockroachDB",
     "Postgres-compatible distributed database. Auth: paste connection string OR host + port (default 26257) + database + user + password.",
     "database"),
    ("neo4j", "Neo4j",
     "Graph database. Auth: paste connection URI (bolt:// or neo4j://) + username + password + optional database name.",
     "database"),
    ("couchbase", "Couchbase",
     "Distributed document database. Auth: paste connection string (couchbase:// or couchbases://) + username + password + bucket.",
     "database"),
    ("dynamodb", "DynamoDB",
     "AWS managed NoSQL. Auth: paste AWS Access Key ID + Secret Access Key + region. Same shape as the AWS connector but scoped to DynamoDB.",
     "database"),
    ("firestore", "Firestore",
     "Google Cloud document database. Auth: paste service account JSON (textarea, secret) + GCP project ID. Same shape as BigQuery.",
     "database"),
    ("influxdb", "InfluxDB",
     "Time-series database. Auth: paste URL + token + organization + bucket.",
     "database"),

    # ── Vector databases ──
    ("pinecone", "Pinecone",
     "Managed vector database. Auth: paste API key + environment (e.g. us-west1-gcp) + index name.",
     "cube"),
    ("weaviate", "Weaviate",
     "Open-source + managed vector database. Auth: paste cluster URL + API key.",
     "cube"),
    ("qdrant", "Qdrant",
     "Open-source + managed vector database. Auth: paste cluster URL + API key.",
     "cube"),
    ("chroma", "Chroma",
     "Open-source vector database. Auth: paste server URL + optional API token (for hosted Chroma Cloud).",
     "cube"),

    # ── Cloud providers ──
    ("aws", "AWS",
     "Amazon Web Services (S3, EC2, Lambda, etc.). Auth: paste Access Key ID + Secret Access Key + default region. Optional session token for STS-issued temporary credentials.",
     "cube"),
    ("gcp", "Google Cloud",
     "Google Cloud Platform (GCS, Compute Engine, etc.). Auth: paste service account JSON (textarea, secret) + project ID. Same service-account shape as BigQuery/Drive.",
     "cube"),
    ("azure", "Azure",
     "Microsoft Azure. Auth: paste tenant ID + client ID (app registration) + client secret (service principal). Optional subscription ID.",
     "cube"),
    ("cloudflare", "Cloudflare",
     "DNS, Workers, R2, KV, etc. Auth: paste API token (scoped) from My Profile → API Tokens. Account ID is required for most resources.",
     "cube"),
    ("digitalocean", "DigitalOcean",
     "DO droplets, Spaces, App Platform, etc. Auth: paste personal access token from API → Tokens.",
     "cube"),
    ("heroku", "Heroku",
     "PaaS for apps. Auth: paste API key from Account settings → API Key.",
     "cube"),
    ("render", "Render",
     "Modern PaaS. Auth: paste API key from Account settings → API Keys.",
     "cube"),
    ("fly_io", "Fly.io",
     "Edge app platform. Auth: paste API token (from `flyctl auth token` or Account → Access Tokens).",
     "cube"),

    # ── Database variants — same wire protocol as the core engines,
    # different vendor consoles and connection-string conventions.
    # The category is `database` so they live in the unified Databases
    # section alongside Postgres / MySQL / Snowflake / etc.

    # Postgres variants
    ("neon", "Neon",
     "Serverless Postgres with branching. Auth: paste connection string from Neon Console → Project → Connection Details (or host + port + database + user + password). Category: database.",
     "database"),
    ("supabase", "Supabase",
     "Managed Postgres + Auth/Storage/Realtime APIs. Auth: paste connection string from Project Settings → Database → Connection string (or host + port + db + user + password). Category: database.",
     "database"),
    ("aws_rds_postgres", "AWS RDS Postgres",
     "AWS-managed Postgres. Auth: paste host (RDS endpoint) + port (default 5432) + database + user + password. SSL toggle. Category: database. Walk through finding the endpoint in AWS Console → RDS → Databases.",
     "database"),
    ("aws_aurora_postgres", "AWS Aurora Postgres",
     "AWS Aurora cluster (Postgres-compatible). Auth: paste cluster endpoint host + port (default 5432) + database + user + password. SSL toggle. Category: database. Walk through AWS Console → RDS → Databases (Aurora cluster). Distinguish writer vs reader endpoint.",
     "database"),
    ("gcp_cloudsql_postgres", "GCP Cloud SQL Postgres",
     "Google Cloud Postgres. Auth: paste public IP (or Cloud SQL Auth Proxy host) + port (default 5432) + database + user + password. SSL toggle. Category: database. Walk through GCP Console → Cloud SQL → instances.",
     "database"),
    ("azure_postgres", "Azure Postgres",
     "Azure Database for PostgreSQL (Flexible Server). Auth: paste host (yourdb.postgres.database.azure.com) + port (default 5432) + database + user + password. SSL required. Category: database. Walk through Azure Portal → Azure Database for PostgreSQL.",
     "database"),
    ("heroku_postgres", "Heroku Postgres",
     "Heroku-managed Postgres add-on. Auth: paste DATABASE_URL connection string from Heroku Dashboard → app → Resources → Heroku Postgres → Settings → View Credentials. SSL required. Category: database.",
     "database"),

    # MySQL variants
    ("planetscale", "PlanetScale",
     "MySQL-compatible serverless platform. Auth: paste host + port (default 3306) + database + username + password from PlanetScale Console → Branch → Connect → New password. SSL required. Category: database.",
     "database"),
    ("aws_rds_mysql", "AWS RDS MySQL",
     "AWS-managed MySQL. Auth: paste host (RDS endpoint) + port (default 3306) + database + user + password. SSL toggle. Category: database. Walk through AWS Console → RDS → Databases.",
     "database"),
    ("aws_aurora_mysql", "AWS Aurora MySQL",
     "AWS Aurora cluster (MySQL-compatible). Auth: paste cluster endpoint + port (default 3306) + database + user + password. SSL toggle. Category: database. Distinguish writer vs reader endpoint.",
     "database"),
    ("gcp_cloudsql_mysql", "GCP Cloud SQL MySQL",
     "Google Cloud MySQL. Auth: paste public IP + port (default 3306) + database + user + password. SSL toggle. Category: database.",
     "database"),
    ("azure_mysql", "Azure MySQL",
     "Azure Database for MySQL (Flexible Server). Auth: paste host + port (default 3306) + database + user + password. SSL required. Category: database.",
     "database"),

    # SQL Server variants
    ("azure_sql", "Azure SQL Database",
     "Microsoft-managed SQL Server in Azure. Auth: paste host (yourdb.database.windows.net) + port (default 1433) + database + user + password. Encryption required. Category: database. Walk through Azure Portal → SQL databases.",
     "database"),
    ("aws_rds_sqlserver", "AWS RDS SQL Server",
     "AWS-managed SQL Server. Auth: paste host (RDS endpoint) + port (default 1433) + database + user + password. Encryption toggle. Category: database.",
     "database"),

    # MongoDB variants
    ("mongodb_atlas", "MongoDB Atlas",
     "MongoDB-managed cloud. Auth: paste connection string (mongodb+srv://) from Atlas Console → Database → Connect → Connect your application. Don't forget to add your IP to Atlas → Network Access. Category: database.",
     "database"),

    # Cloud-native NoSQL / SQL
    ("azure_cosmos_db", "Azure Cosmos DB",
     "Microsoft globally-distributed multi-model database. Auth: paste account endpoint URI + primary key from Azure Portal → Cosmos DB → Keys. Optional database + container name. Category: database.",
     "database"),
    ("gcp_spanner", "Google Spanner",
     "Google globally-distributed SQL database. Auth: paste service account JSON (textarea, secret) + project ID + instance ID + database ID. Same service-account shape as BigQuery. Category: database.",
     "database"),

    # Redis variants
    ("upstash", "Upstash",
     "Serverless Redis (and Kafka). Auth: paste REST URL + REST token from Upstash Console → Database → REST API. Or paste connection string for the standard Redis protocol. Category: database.",
     "database"),

    # ── Observability / monitoring ──
    ("grafana", "Grafana",
     "Dashboards + alerting. Auth: paste service account token (or API key on older instances) + Grafana URL (cloud: <stack>.grafana.net or self-hosted). Category: observability.",
     "sparkle"),
    ("prometheus", "Prometheus",
     "Open-source metrics + time-series. Auth: paste server URL + optional basic auth (user + password) for protected instances. Category: observability.",
     "sparkle"),
    ("honeycomb", "Honeycomb",
     "Tracing + observability for high-cardinality data. Auth: paste API key from Account → Team Settings → API Keys. Category: observability.",
     "sparkle"),
    ("splunk", "Splunk",
     "Logs + security analytics. Auth: paste instance URL + HEC (HTTP Event Collector) token, OR auth token from Settings → Tokens. Category: observability.",
     "sparkle"),
    ("dynatrace", "Dynatrace",
     "Full-stack observability + APM. Auth: paste API token (with required scopes) + environment URL (e.g. https://abc12345.live.dynatrace.com). Category: observability.",
     "sparkle"),
    ("sumo_logic", "Sumo Logic",
     "Logs + metrics + security analytics. Auth: paste Access ID + Access Key from Preferences → Access Keys. Region selector (us1|us2|eu|au|jp|ca|de|in). Category: observability.",
     "sparkle"),
    ("better_stack", "Better Stack",
     "Logs + uptime + on-call (formerly Logtail + Better Uptime). Auth: paste team API token from Account settings → API tokens. Category: observability.",
     "sparkle"),
    ("statuspage", "Statuspage",
     "Atlassian public/private status pages. Auth: paste API key from User Profile → API. Category: observability.",
     "sparkle"),
    ("pingdom", "Pingdom",
     "Uptime + synthetic monitoring (SolarWinds). Auth: paste API token from Settings → Pingdom API. Category: observability.",
     "sparkle"),
    ("uptimerobot", "UptimeRobot",
     "Simple uptime monitoring. Auth: paste main API key OR monitor-specific read-only API key from My Settings → API Settings. Category: observability.",
     "sparkle"),
    ("bugsnag", "BugSnag",
     "Error monitoring (SmartBear). Auth: paste personal auth token from My Account → Settings → Personal auth tokens. Category: observability.",
     "brain"),
    ("rollbar", "Rollbar",
     "Error monitoring. Auth: paste project access token from Project → Settings → Project Access Tokens. Category: observability.",
     "brain"),

    # ── Accounting ──
    ("xero", "Xero",
     "Accounting platform (UK/AU/NZ). Auth: OAuth 2.0 BYOK. auth_url=https://login.xero.com/identity/connect/authorize, token_url=https://identity.xero.com/connect/token. Scopes: accounting.transactions, accounting.contacts, offline_access. Category: accounting.",
     "key"),
    ("sage_intacct", "Sage Intacct",
     "Sage's mid-market financial management product. Auth: paste sender ID + sender password + company ID + user ID + user password (Sage's XML web services use multi-credential authentication). Category: accounting.",
     "key"),
    ("freshbooks", "FreshBooks",
     "Cloud accounting for small businesses. Auth: OAuth 2.0 BYOK. auth_url=https://auth.freshbooks.com/oauth/authorize, token_url=https://api.freshbooks.com/auth/oauth/token. Category: accounting.",
     "key"),
    ("netsuite", "NetSuite",
     "Oracle ERP/financials. Auth: Token-Based Authentication (TBA) — paste account ID + consumer key + consumer secret + token ID + token secret. Five-field paste, all secret. Category: accounting.",
     "key"),
    ("zoho_books", "Zoho Books",
     "Zoho's accounting product. Auth: OAuth 2.0 BYOK (app from api-console.zoho.com). auth_url=https://accounts.zoho.com/oauth/v2/auth, token_url=https://accounts.zoho.com/oauth/v2/token. Scopes: ZohoBooks.fullaccess.all. Region selector (com|eu|in|com.au|jp). Category: accounting.",
     "key"),

    # ── HR / People Ops ──
    ("fifteen_five", "15Five",
     "Performance management + check-ins + OKRs. Auth: paste personal API token from Account Settings → Integrations → API. Category: hr.",
     "check"),
    ("lattice", "Lattice",
     "Performance + engagement + 1:1s. Auth: paste API key from Admin → Integrations → API. Category: hr.",
     "check"),
    ("bamboohr", "BambooHR",
     "HRIS for SMB. Auth: paste API key + company subdomain (yourcompany.bamboohr.com). API key is generated from the user's profile → API Keys. Category: hr.",
     "check"),
    ("gusto", "Gusto",
     "Payroll + benefits + HR. Auth: OAuth 2.0 BYOK (app from dev.gusto.com). auth_url=https://api.gusto.com/oauth/authorize, token_url=https://api.gusto.com/oauth/token. Scopes: read+write per resource. Category: hr.",
     "check"),
    ("rippling_hris", "Rippling HR",
     "Rippling's HRIS API. Auth: OAuth 2.0 BYOK (app from app.rippling.com → Settings → API). auth_url=https://app.rippling.com/apps/oauth/authorize, token_url=https://app.rippling.com/api/o/token/. Category: hr.",
     "check"),
    ("deel", "Deel",
     "Global hiring + payroll for contractors + EOR. Auth: paste API key from Settings → Developer → API. Category: hr.",
     "check"),
    ("hibob", "HiBob",
     "Modern HRIS. Auth: paste service user token from Settings → Integrations → Service users. Category: hr.",
     "check"),
    ("personio", "Personio",
     "European HRIS. Auth: paste client ID + client secret from Settings → Integrations → API. Category: hr.",
     "check"),

    # ── Recruiting / ATS ──
    ("greenhouse", "Greenhouse",
     "Applicant tracking + recruiting. Auth: paste Harvest API key from Configure → Dev Center → API Credential Management. Category: hr.",
     "check"),
    ("lever", "Lever",
     "Applicant tracking. Auth: paste API key from Settings → Integrations → API. Category: hr.",
     "check"),
    ("ashby", "Ashby",
     "Applicant tracking. Auth: paste API key from Admin → Integrations → API. Category: hr.",
     "check"),
    ("workable", "Workable",
     "Applicant tracking. Auth: paste API token + subdomain (yourcompany.workable.com) from Settings → Integrations → Access Tokens. Category: hr.",
     "check"),

    # ── Web search APIs ──
    ("exa", "Exa",
     "Neural search built for AI agents. Auth: paste API key from dashboard.exa.ai → API Keys. Category: web-search.",
     "search"),
    ("tavily", "Tavily",
     "Search API tuned for AI agents (results pre-formatted for LLM consumption). Auth: paste API key (tvly-…) from app.tavily.com → API Keys. Category: web-search.",
     "search"),
    ("google_search", "Google Search",
     "Google Programmable Search Engine (Custom Search JSON API). Auth: paste API key from Google Cloud Console (Custom Search API enabled) + Search Engine ID (cx) from programmablesearchengine.google.com. Category: web-search.",
     "search"),
    ("bing_search", "Bing Search",
     "Microsoft Bing Web Search v7 (via Azure). Auth: paste subscription key from Azure Portal → Bing Search resource → Keys and Endpoint. Optional Azure region. Category: web-search.",
     "search"),
    ("brave_search", "Brave Search",
     "Independent search index. Auth: paste API key from api.search.brave.com → Subscriptions. Category: web-search.",
     "search"),
    ("serper", "Serper",
     "Real-time Google search results API. Auth: paste API key from serper.dev → API Key. Category: web-search.",
     "search"),
    ("serpapi", "SerpAPI",
     "Scraped Google + Bing + Yahoo + Baidu search results. Auth: paste API key from serpapi.com → API Key. Category: web-search.",
     "search"),
    ("perplexity", "Perplexity",
     "Perplexity's search-grounded LLM API. Auth: paste API key (pplx-…) from perplexity.ai → Settings → API. Category: web-search.",
     "search"),
    ("you_com", "You.com",
     "You.com web + image + news search API. Auth: paste API key from api.you.com → Dashboard. Category: web-search.",
     "search"),
    ("kagi", "Kagi",
     "Premium ad-free search with paid API. Auth: paste API key from kagi.com → Settings → API. Category: web-search.",
     "search"),

    # ── Maps / Geocoding / Routing ──
    ("google_maps", "Google Maps Platform",
     "Places + Geocoding + Routes + Distance Matrix. Auth: paste API key from Google Cloud Console → APIs & Services → Credentials (with Maps APIs enabled). Optional API restrictions. Category: maps.",
     "globe"),
    ("mapbox", "Mapbox",
     "Maps + geocoding + routing + static images. Auth: paste access token (pk.eyJ…) from account.mapbox.com → Access Tokens. Category: maps.",
     "globe"),
    ("here_maps", "HERE Maps",
     "Enterprise maps + geocoding + routing. Auth: paste API key from developer.here.com → Projects → REST. Category: maps.",
     "globe"),
    ("tomtom", "TomTom",
     "Maps + geocoding + traffic + routing. Auth: paste API key from developer.tomtom.com → My Dashboard → Keys & Usage. Category: maps.",
     "globe"),
    ("openstreetmap", "OpenStreetMap",
     "Free open-data maps via Nominatim (geocoding) + Overpass (queries). Auth: optional — no key needed for low volume; paste a contact email so OSM can reach you about heavy use. Category: maps.",
     "globe"),

    # ── Mobility / Rideshare / Delivery ──
    ("uber", "Uber",
     "Uber Rides + Eats + Business. Auth: OAuth 2.0 BYOK (app from developer.uber.com). auth_url=https://login.uber.com/oauth/v2/authorize, token_url=https://login.uber.com/oauth/v2/token. Scopes: profile, history, rides.request, rides.read, eats.deliveries (subset depending on use). Category: mobility.",
     "send"),
    ("lyft", "Lyft",
     "Lyft Rides + Business. Auth: OAuth 2.0 BYOK (app from www.lyft.com/developers). auth_url=https://api.lyft.com/oauth/authorize, token_url=https://api.lyft.com/oauth/token. Scopes: public, profile, rides.request, rides.read. Category: mobility.",
     "send"),
    ("doordash", "DoorDash",
     "DoorDash Drive (delivery-as-a-service) + Marketplace. Auth: paste developer key + signing secret from developer.doordash.com → Drive Portal → Credentials. Anton signs each request with HS256 JWT. Category: mobility.",
     "send"),
    ("instacart", "Instacart",
     "Instacart Connect (retail / partner API). Auth: paste API key + retailer ID issued by your Instacart partner team. Not self-serve — coordinate with Instacart Connect onboarding. Category: mobility.",
     "send"),
    ("bolt", "Bolt",
     "Bolt rides + food + scooters (EU/Africa). Auth: paste API key + secret from partners.bolt.eu → Developers. Category: mobility.",
     "send"),
    ("grab", "Grab",
     "Grab rides + food + financial services (SE Asia). Auth: OAuth 2.0 BYOK (app from developer.grab.com). auth_url=https://api.grab.com/grabid/v1/oauth2/authorize, token_url=https://api.grab.com/grabid/v1/oauth2/token. Category: mobility.",
     "send"),

    # ── AI APIs (LLMs / voice / image) ──
    ("openai", "OpenAI",
     "GPT models + embeddings + DALL-E + Whisper. Auth: paste API key (sk-…) from platform.openai.com → API Keys. Optional organization ID + project ID. Category: ai.",
     "brain"),
    ("anthropic", "Anthropic",
     "Claude models. Auth: paste API key (sk-ant-…) from console.anthropic.com → API Keys. Category: ai.",
     "brain"),
    ("google_gemini", "Google Gemini",
     "Google's Gemini API (consumer-grade, distinct from Vertex AI). Auth: paste API key from aistudio.google.com → Get API key. Category: ai.",
     "brain"),
    ("cohere", "Cohere",
     "Command models + embeddings + reranking. Auth: paste API key from dashboard.cohere.com → API Keys. Category: ai.",
     "brain"),
    ("mistral", "Mistral",
     "Mistral's hosted models. Auth: paste API key from console.mistral.ai → API Keys. Category: ai.",
     "brain"),
    ("huggingface", "Hugging Face",
     "Inference API + model hub access. Auth: paste user access token (hf_…) from huggingface.co → Settings → Access Tokens. Pick read or write scope. Category: ai.",
     "brain"),
    ("replicate", "Replicate",
     "Run open-source models via API. Auth: paste API token (r8_…) from replicate.com → Account. Category: ai.",
     "brain"),
    ("together_ai", "Together.ai",
     "Inference + fine-tuning for open-source models. Auth: paste API key from api.together.xyz → Settings → API Keys. Category: ai.",
     "brain"),
    ("groq", "Groq",
     "Ultra-low-latency inference. Auth: paste API key (gsk_…) from console.groq.com → API Keys. Category: ai.",
     "brain"),
    ("fireworks", "Fireworks AI",
     "Hosted open-source models. Auth: paste API key from fireworks.ai → Account → API Keys. Category: ai.",
     "brain"),
    ("elevenlabs", "ElevenLabs",
     "AI voice synthesis + cloning. Auth: paste API key (xi_…) from elevenlabs.io → Profile → API Key. Category: ai.",
     "mic"),
    ("deepgram", "Deepgram",
     "Speech-to-text + voice intelligence. Auth: paste API key from console.deepgram.com → API Keys. Category: ai.",
     "mic"),
    ("assemblyai", "AssemblyAI",
     "Speech-to-text + audio intelligence. Auth: paste API key from www.assemblyai.com → Account → API Keys. Category: ai.",
     "mic"),
    ("stability_ai", "Stability AI",
     "Stable Diffusion image generation API. Auth: paste API key (sk-…) from platform.stability.ai → Account → API Keys. Category: ai.",
     "image"),
    ("runway", "Runway",
     "AI video generation (Gen-2 / Gen-3). Auth: paste API key from dev.runwayml.com → API Keys. Category: ai.",
     "image"),
    ("pika", "Pika",
     "AI video generation. Auth: paste API key (when generally available — currently waitlisted) from pika.art → Developers. Category: ai.",
     "image"),
    ("luma", "Luma AI",
     "Dream Machine video + Genie 3D generation. Auth: paste API key from lumalabs.ai → Dream Machine → API. Category: ai.",
     "image"),

    # ── Public data APIs ──
    ("newsapi", "NewsAPI",
     "Aggregated news headlines + articles. Auth: paste API key from newsapi.org → Account → API Key. Category: public-data.",
     "doc"),
    ("openweather", "OpenWeather",
     "Weather forecasts + historical data. Auth: paste API key from openweathermap.org → API Keys. Category: public-data.",
     "sun"),
    ("tomorrow_io", "Tomorrow.io",
     "Hyperlocal weather + climate intelligence. Auth: paste API key from app.tomorrow.io → Settings → API Keys. Category: public-data.",
     "sun"),
    ("accuweather", "AccuWeather",
     "Weather forecasts + alerts. Auth: paste API key from developer.accuweather.com → My Apps. Category: public-data.",
     "sun"),
    ("alphavantage", "Alpha Vantage",
     "Stocks + forex + crypto + economic indicators. Auth: paste API key from www.alphavantage.co → Get Free API Key. Category: public-data.",
     "sparkle"),
    ("polygon_io", "Polygon.io",
     "Stocks + options + forex + crypto market data. Auth: paste API key from polygon.io → Dashboard → Keys. Category: public-data.",
     "sparkle"),
    ("coingecko", "CoinGecko",
     "Crypto prices + market data + on-chain analytics. Auth: paste API key from www.coingecko.com → Developer Dashboard → Pro API Keys (free tier requires no key but rate-limited). Category: public-data.",
     "cube"),
    ("etherscan", "Etherscan",
     "Ethereum blockchain explorer + smart-contract data. Auth: paste API key from etherscan.io → My API Keys. Category: public-data.",
     "cube"),
    ("youtube_data", "YouTube Data",
     "Search + read videos, channels, playlists, comments. Auth: OAuth 2.0 BYOK Google Cloud client (auth_url=https://accounts.google.com/o/oauth2/v2/auth, token_url=https://oauth2.googleapis.com/token, scopes: youtube.readonly + youtube.force-ssl + userinfo.email; extra_auth_params {access_type:offline, prompt:consent}). Category: public-data.",
     "image"),
    ("spotify", "Spotify",
     "Music catalog + user library + playlists. Auth: OAuth 2.0 BYOK (app from developer.spotify.com/dashboard). auth_url=https://accounts.spotify.com/authorize, token_url=https://accounts.spotify.com/api/token. Common scopes: user-library-read, playlist-read-private, user-read-recently-played. Category: public-data.",
     "image"),

    # ── Logistics / Shipping ──
    ("shipstation", "ShipStation",
     "Multi-carrier shipping + label printing for ecommerce. Auth: paste API key + API secret from Account Settings → API Settings. Category: logistics.",
     "upload"),
    ("shippo", "Shippo",
     "Multi-carrier shipping API. Auth: paste API token (live_… or test_…) from goshippo.com → Settings → API. Category: logistics.",
     "upload"),
    ("easypost", "EasyPost",
     "Multi-carrier shipping + tracking + insurance. Auth: paste API key from www.easypost.com → Account Settings → API Keys. Test vs Production keys distinguished by prefix. Category: logistics.",
     "upload"),
    ("shipbob", "ShipBob",
     "3PL fulfillment + warehouse + shipping. Auth: paste personal access token from web.shipbob.com → Integrations → API. Category: logistics.",
     "upload"),

    # ── OpenTelemetry & observability stack additions ──
    ("otel_collector", "OpenTelemetry Collector",
     "OTLP endpoint for traces / metrics / logs (vendor-neutral). Auth: paste collector OTLP endpoint URL (gRPC or HTTP) + optional bearer token + optional headers. Category: observability.",
     "sparkle"),
    ("jaeger", "Jaeger",
     "Distributed tracing (CNCF). Auth: paste Jaeger query URL + optional basic auth (user + password) for protected instances. Category: observability.",
     "sparkle"),
    ("zipkin", "Zipkin",
     "Distributed tracing (Twitter). Auth: paste Zipkin server URL + optional API token for hosted instances. Category: observability.",
     "sparkle"),
    ("loki", "Grafana Loki",
     "Log aggregation (Grafana). Auth: paste Loki URL + optional basic auth (user + password) or X-Scope-OrgID for multi-tenant setups. Category: observability.",
     "sparkle"),
    ("tempo", "Grafana Tempo",
     "Tracing backend (Grafana). Auth: paste Tempo URL + optional basic auth + X-Scope-OrgID. Category: observability.",
     "sparkle"),
    ("victoriametrics", "VictoriaMetrics",
     "High-performance Prometheus-compatible metrics + logs. Auth: paste VictoriaMetrics URL + optional bearer token / basic auth. Category: observability.",
     "sparkle"),
    ("aws_cloudwatch", "AWS CloudWatch",
     "AWS-native metrics + logs + alarms. Auth: paste AWS Access Key ID + Secret Access Key + region. Same shape as the AWS connector but scoped to CloudWatch. Category: observability.",
     "sparkle"),
    ("gcp_cloud_monitoring", "GCP Cloud Monitoring",
     "Google Cloud's metrics + logs (formerly Stackdriver). Auth: paste service account JSON (textarea, secret) + project ID. Category: observability.",
     "sparkle"),
    ("azure_monitor", "Azure Monitor",
     "Azure-native metrics + logs + Application Insights. Auth: paste tenant ID + client ID + client secret (service principal) + subscription ID. Category: observability.",
     "sparkle"),
    ("elastic_apm", "Elastic APM",
     "Elastic's APM + tracing. Auth: paste APM Server URL + secret token (or API key) from Kibana → Stack Management → Fleet → APM Integration. Category: observability.",
     "search"),
    ("graylog", "Graylog",
     "Open-source log management. Auth: paste Graylog API URL + access token (or username + password) from System → Authentication → Tokens. Category: observability.",
     "list"),
    ("logz_io", "Logz.io",
     "Hosted ELK + Prometheus + APM. Auth: paste shipping token (for log ingest) + API token (for queries) + region (us|eu|au|ca|wa). Category: observability.",
     "sparkle"),
    ("coralogix", "Coralogix",
     "Logs + metrics + tracing platform. Auth: paste API key + region domain (e.g. coralogix.com, eu2.coralogix.com, app.coralogix.in). Category: observability.",
     "sparkle"),
    ("checkly", "Checkly",
     "Synthetic monitoring + API monitoring + e2e Playwright. Auth: paste API key from app.checklyhq.com → Account Settings → API Keys + account ID. Category: observability.",
     "refresh"),
    ("appdynamics", "AppDynamics",
     "Cisco APM. Auth: paste controller URL + API client name + API client secret from Settings → API Clients. Category: observability.",
     "sparkle"),
    ("instana", "Instana",
     "IBM observability + APM. Auth: paste base URL (e.g. unit-tenant.instana.io) + API token from Settings → Team Settings → API Tokens. Category: observability.",
     "sparkle"),
]


# ─── Prompt assembly ──────────────────────────────────────────────────


def build_prompt(target_id: str, label: str, hint: str, suggested_logo: str, example: str) -> tuple[str, str]:
    """Returns (system_prompt, user_message)."""
    system = f"""You are generating connector spec JSON files for an Electron desktop app called Anton.

The app shows a connector picker; clicking a connector opens a form. The form spec is a JSON
object the renderer (DataVaultForm.jsx) consumes directly. Your job: produce one such JSON
object for a given connector.

# Schema

```
{{
  "id": "<slug>",                         // matches the filename
  "label": "<Display Name>",
  "aliases": ["<alt name>", ...],         // for fuzzy matching
  "keywords": ["<word>", ...],            // for token-overlap scoring
  "description": "<one sentence>",
  "category": "<one of: crm | sales-engagement | enrichment | marketing | analytics | ads | support | customer-success | revenue-intel | communication | data-warehouse | data | scheduling | forms | documents | billing | productivity | files | other>",
  "logo": "<one of the icon names from LOGOS list below>",
  "logo_color": "<hex like #FF7A59>",
  "form": {{
    "form_id": "<slug>-connector",
    "title": "Connect <Label>",
    "subtitle": "<one sentence>",
    "logo": "<same as top-level logo>",
    "logo_color": "<same hex>",
    "methods": [
      {{
        "id": "<method-id-kebab>",
        "label": "<Method Display Name>",
        "description": "<1-2 sentences shown on the picker card>",
        "recommended": <true|false>,       // exactly one method should be recommended
        "how_to": "<markdown setup walkthrough — multi-line, ## headings ok>",
        "help_url": "<external help URL>",

        // For OAuth methods only:
        "submit_action": "oauth_launch",
        "oauth": {{
          "auth_url": "<authorization URL>",
          "token_url": "<token endpoint>",
          "scopes": ["..."],
          "extra_auth_params": {{...}}      // optional
        }},

        // Always:
        "fields": [
          {{
            "name": "<field-name-snake>",
            "label": "<Field Label>",
            "type": "text" | "password" | "url" | "select" | "textarea" | "boolean",
            "required": true | false,
            "secret": true,                  // for password/textarea fields holding credentials
            "placeholder": "<hint>",
            "default": "<default value>",
            "description": "<helper text under the input>",
            "options": [{{"value": "...", "label": "..."}}]   // for select only
          }}
        ]
      }}
    ]
  }}
}}
```

# Available logo names (pick the closest semantic match)

{', '.join(LOGOS)}

# Recommended-method rule (CRITICAL)

Exactly one method has `"recommended": true`. The recommended method MUST be the SIMPLEST
copy-paste path the user can take. OAuth is recommended ONLY when there's no simpler path
(no API key, no personal access token, no app password). Paste-an-API-key always beats OAuth
when both exist.

# how_to writing rules

- Markdown. Use `## Section` headings.
- Numbered steps for the setup walkthrough.
- Tell the user EXACTLY where in the vendor UI to click (Settings → X → Y).
- Mention any gotchas (token shown once, region differences, scopes that matter, prereqs).
- Keep it tight: 5-15 lines per method. Not an essay.
- Don't include code blocks unless absolutely necessary.

# OAuth flow notes

- All OAuth methods include `"submit_action": "oauth_launch"` and an `"oauth"` block.
- Pattern A (hosted client) means the spec ships a `client_id` baked in — don't do that here, we don't ship hosted clients yet.
- Pattern B (BYOK) means the user fills in `client_id` + `client_secret` fields. Always include those two fields for OAuth methods (text + password types).
- For Google services: extra_auth_params={{"access_type": "offline", "prompt": "consent"}} so refresh tokens issue.

# Field shape quick reference

- `text`: single-line text input
- `password`: masked, always set `"secret": true`
- `url`: validated URL input
- `select`: dropdown — must include `options[]`
- `textarea`: multi-line (e.g. JSON keys); set `"secret": true` for credential pastes
- `boolean`: checkbox

# Output requirements

- Output ONE JSON object. NO markdown code fence, NO surrounding prose. Just `{{` to `}}`.
- All required-field values present. No null/undefined.
- Field names in `snake_case`, method ids in `kebab-case`.
- Help URLs must be plausible canonical vendor docs URLs.

# Example: Gmail connector

{example}
"""

    user = f"""Generate the connector JSON for:

- **id**: {target_id}
- **label**: {label}
- **suggested logo**: {suggested_logo}
- **hint**: {hint}

Use the hint to drive the methods + auth fields. Apply the recommended-method rule
strictly. Output JSON only.
"""
    return system, user


# ─── LLM client (Anthropic or OpenAI-compatible) ─────────────────────


def _load_dotenv_once():
    """Pull values from ~/.anton/.env if env vars aren't already set —
    that's where the desktop app stores them, so the generator inherits
    the same provider config Anton itself uses."""
    candidates = [
        Path.home() / ".anton" / ".env",
        OUT_DIR.parents[2] / ".env",
    ]
    for p in candidates:
        if not p.is_file():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v


def make_client():
    """Returns a callable `(system, user) -> str` that hits whichever
    provider is configured. Prefers Anthropic if an API key is present;
    falls back to the OpenAI-compatible endpoint Anton uses (which on
    this machine is a MindsDB proxy)."""
    _load_dotenv_once()

    # ── Anthropic native ──
    a_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTON_ANTHROPIC_API_KEY")
    if a_key:
        model = os.environ.get("ANTON_PLANNING_MODEL", DEFAULT_ANTHROPIC_MODEL)
        # Sanity: only forward to Anthropic if the model name looks like a Claude model.
        if not model.startswith("claude"):
            model = DEFAULT_ANTHROPIC_MODEL
        print(f"[provider] Anthropic ({model})")
        return _anthropic_caller(a_key, model)

    # ── OpenAI-compatible (Anton's existing config — likely MindsDB) ──
    o_key = os.environ.get("ANTON_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    o_base = os.environ.get("ANTON_OPENAI_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
    o_model = os.environ.get("ANTON_PLANNING_MODEL") or os.environ.get("OPENAI_MODEL")
    if o_key and o_base and o_model:
        print(f"[provider] OpenAI-compatible ({o_model} @ {o_base})")
        return _openai_caller(o_key, o_base.rstrip("/"), o_model)

    raise SystemExit(
        "No provider configured. Either set ANTHROPIC_API_KEY, or set "
        "ANTON_OPENAI_API_KEY + ANTON_OPENAI_BASE_URL + ANTON_PLANNING_MODEL "
        "in your environment or ~/.anton/.env."
    )


def _anthropic_caller(api_key: str, model: str):
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    def call(system: str, user: str, retries: int = 2) -> str:
        body = {
            "model": model,
            "max_tokens": MAX_TOKENS,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        last_err = None
        for attempt in range(retries + 1):
            try:
                with httpx.Client(timeout=180.0) as client:
                    r = client.post("https://api.anthropic.com/v1/messages",
                                    headers=headers, json=body)
                if r.status_code == 200:
                    payload = r.json()
                    parts = [b.get("text", "") for b in payload.get("content", []) if b.get("type") == "text"]
                    return "".join(parts).strip()
                if r.status_code in (429, 529, 500, 502, 503, 504):
                    last_err = f"HTTP {r.status_code}: {r.text[:200]}"
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError(f"Anthropic API error {r.status_code}: {r.text[:500]}")
            except httpx.RequestError as e:
                last_err = str(e)
                time.sleep(2 ** attempt)
        raise RuntimeError(f"Anthropic call failed: {last_err}")

    return call


def _openai_caller(api_key: str, base_url: str, model: str):
    """Hits the OpenAI Chat Completions API (or any compatible proxy
    like MindsDB)."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "content-type": "application/json",
    }

    def call(system: str, user: str, retries: int = 2) -> str:
        body = {
            "model": model,
            "max_tokens": MAX_TOKENS,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            # `response_format: json_object` is supported by OpenAI and
            # most compatibles. If a proxy ignores it, the system
            # prompt's "JSON only" instruction still steers the output.
            "response_format": {"type": "json_object"},
        }
        last_err = None
        for attempt in range(retries + 1):
            try:
                with httpx.Client(timeout=180.0) as client:
                    r = client.post(f"{base_url}/chat/completions",
                                    headers=headers, json=body)
                if r.status_code == 200:
                    payload = r.json()
                    choices = payload.get("choices") or []
                    if not choices:
                        raise RuntimeError(f"No choices in OpenAI response: {payload}")
                    return (choices[0].get("message") or {}).get("content", "").strip()
                if r.status_code in (429, 500, 502, 503, 504):
                    last_err = f"HTTP {r.status_code}: {r.text[:200]}"
                    time.sleep(2 ** attempt)
                    continue
                # Some proxies reject response_format — drop it and retry once.
                if r.status_code == 400 and "response_format" in r.text and "response_format" in body:
                    body.pop("response_format", None)
                    continue
                raise RuntimeError(f"OpenAI API error {r.status_code}: {r.text[:500]}")
            except httpx.RequestError as e:
                last_err = str(e)
                time.sleep(2 ** attempt)
        raise RuntimeError(f"OpenAI call failed: {last_err}")

    return call


# ─── Validation ───────────────────────────────────────────────────────


def parse_and_validate(text: str, expected_id: str) -> dict:
    """Parse JSON. Strip any accidental markdown fences. Validate the
    minimum shape the registry expects. Raise on failure with a
    description the LLM can use to retry."""
    cleaned = text.strip()
    # Defensive — sometimes models still emit a fence despite instructions.
    if cleaned.startswith("```"):
        # Drop first line and last fence.
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    try:
        spec = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"Output is not valid JSON: {e}") from e

    # Required top-level keys
    for key in ("id", "label", "category", "logo", "logo_color", "form"):
        if key not in spec:
            raise ValueError(f"Missing top-level key: `{key}`")
    if spec["id"] != expected_id:
        raise ValueError(f"id mismatch: expected `{expected_id}`, got `{spec['id']}`")
    if spec["logo"] not in LOGOS:
        raise ValueError(f"logo `{spec['logo']}` is not in the allowed palette ({len(LOGOS)} options)")

    form = spec["form"]
    for key in ("form_id", "title", "methods"):
        if key not in form:
            raise ValueError(f"form missing key: `{key}`")
    methods = form["methods"]
    if not isinstance(methods, list) or not methods:
        raise ValueError("form.methods must be a non-empty list")

    recommended_count = 0
    for m in methods:
        if not isinstance(m, dict):
            raise ValueError("each method must be an object")
        for key in ("id", "label", "fields"):
            if key not in m:
                raise ValueError(f"method `{m.get('id', '?')}` missing key: `{key}`")
        if m.get("recommended"):
            recommended_count += 1
        if m.get("submit_action") == "oauth_launch":
            o = m.get("oauth")
            if not isinstance(o, dict):
                raise ValueError(f"method `{m['id']}` has submit_action=oauth_launch but no oauth block")
            for ok in ("auth_url", "token_url", "scopes"):
                if ok not in o:
                    raise ValueError(f"method `{m['id']}` oauth block missing `{ok}`")
    if recommended_count != 1:
        raise ValueError(f"exactly one method must have recommended:true (found {recommended_count})")

    return spec


# ─── Main ─────────────────────────────────────────────────────────────


def load_example_gmail() -> str:
    return (OUT_DIR / "gmail.json").read_text(encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing JSON files. Protected files are still skipped.")
    parser.add_argument("--only", default="",
                        help="Comma-separated list of connector IDs to generate (default: all).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip the LLM call; just print which targets would be processed.")
    args = parser.parse_args()

    only = {s.strip() for s in args.only.split(",") if s.strip()} if args.only else None

    targets = [t for t in TARGETS if (only is None or t[0] in only)]
    if not targets:
        print("No targets match.")
        return

    if args.dry_run:
        for tid, label, _, logo in targets:
            existing = (OUT_DIR / f"{tid}.json").exists()
            tag = "PROTECTED" if tid in PROTECTED else ("EXISTS" if existing else "NEW")
            print(f"  [{tag:9s}] {tid:24s} {label}  ({logo})")
        return

    call = make_client()
    example = load_example_gmail()

    written, skipped, failed = 0, 0, 0
    for tid, label, hint, logo in targets:
        path = OUT_DIR / f"{tid}.json"
        if tid in PROTECTED:
            print(f"  [protected]  skip {tid}")
            skipped += 1
            continue
        if path.exists() and not args.force:
            print(f"  [exists]     skip {tid} (use --force to overwrite)")
            skipped += 1
            continue

        system, user = build_prompt(tid, label, hint, logo, example)

        last_error = None
        spec = None
        for attempt in range(2):
            try:
                t0 = time.monotonic()
                raw = call(system, user)
                dt = time.monotonic() - t0
                spec = parse_and_validate(raw, tid)
                print(f"  [ok]         {tid:24s} {label}  ({dt:.1f}s)")
                break
            except ValueError as e:
                last_error = str(e)
                # Re-prompt with the validation error appended so the
                # LLM can self-correct on attempt 2.
                user = (
                    f"Generate the connector JSON for:\n\n"
                    f"- **id**: {tid}\n- **label**: {label}\n- **suggested logo**: {logo}\n- **hint**: {hint}\n\n"
                    f"Your previous attempt failed validation: {last_error}\n"
                    f"Output JSON only. Apply the recommended-method rule strictly."
                )
                continue

        if spec is None:
            print(f"  [FAIL]       {tid:24s} {last_error}", file=sys.stderr)
            failed += 1
            continue

        path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
        written += 1

    print()
    print(f"summary: {written} written, {skipped} skipped, {failed} failed")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
