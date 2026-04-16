# Datasource Knowledge

Anton reads this file when connecting data sources. For each source, the YAML
block defines the fields Python collects. The prose below describes auth flows,
common errors, and how to handle OAuth2 — Anton handles those using the scratchpad.

Credentials are injected as `DS_<FIELD_UPPER>` environment variables
before any scratchpad code runs. Never embed raw values in code strings.

---

## PostgreSQL

```yaml
engine: postgres
display_name: PostgreSQL
pip: psycopg2-binary
name_from: database
popular: true
fields:
  - { name: host,     required: true,  secret: false, description: "hostname or IP of your database server" }
  - { name: port,     required: true, secret: false, description: "port number", default: "5432" }
  - { name: database, required: true,  secret: false, description: "name of the database to connect" }
  - { name: user,     required: true,  secret: false, description: "database username" }
  - { name: password, required: true,  secret: true,  description: "database password" }
  - { name: schema,   required: false, secret: false, description: "defaults to public if not set" }
  - { name: ssl,      required: false, secret: false, description: "enable SSL (true/false)" }
test_snippet: |
  import psycopg2, os
  conn = psycopg2.connect(
      host=os.environ['DS_HOST'], port=os.environ.get('DS_PORT','5432'),
      dbname=os.environ['DS_DATABASE'], user=os.environ['DS_USER'],
      password=os.environ['DS_PASSWORD'],
  )
  conn.close()
  print("ok")
```

Common errors: "password authentication failed" → wrong password or user.
"could not connect to server" → wrong host/port or firewall blocking.

---

## MySQL

```yaml
engine: mysql
display_name: MySQL
pip: mysql-connector-python
name_from: database
popular: true
fields:
  - { name: host,     required: true,  secret: false, description: "hostname or IP of your MySQL server" }
  - { name: port,     required: true, secret: false, description: "port number", default: "3306" }
  - { name: database, required: true,  secret: false, description: "database name to connect" }
  - { name: user,     required: true,  secret: false, description: "MySQL username" }
  - { name: password, required: true,  secret: true,  description: "MySQL password" }
  - { name: ssl,      required: false, secret: false, description: "enable SSL (true/false)" }
  - { name: ssl_ca,   required: false, secret: false, description: "path to CA certificate file" }
  - { name: ssl_cert, required: false, secret: false, description: "path to client certificate file" }
  - { name: ssl_key,  required: false, secret: false, description: "path to client private key file" }
test_snippet: |
  import mysql.connector, os
  conn = mysql.connector.connect(
      host=os.environ['DS_HOST'],
      port=int(os.environ.get('DS_PORT', '3306')),
      database=os.environ['DS_DATABASE'],
      user=os.environ['DS_USER'],
      password=os.environ['DS_PASSWORD'],
  )
  conn.close()
  print("ok")
```

---

## Snowflake

```yaml
engine: snowflake
display_name: Snowflake
pip: snowflake-connector-python
popular: true
name_from: [account, database]
auth_method: choice
auth_methods:
  - name: password
    display: "Username / Password"
    fields:
      - { name: account,   required: true,  secret: false, description: "Snowflake account identifier (e.g. xy12345.us-east-1 or orgname-accountname)" }
      - { name: user,      required: true,  secret: false, description: "Snowflake username" }
      - { name: password,  required: true,  secret: true,  description: "Snowflake password" }
      - { name: database,  required: true,  secret: false, description: "database name" }
      - { name: schema,    required: false, secret: false, description: "schema name (defaults to PUBLIC)" }
      - { name: warehouse, required: false, secret: false, description: "warehouse to use for queries" }
      - { name: role,      required: false, secret: false, description: "role to assume" }
  - name: key_pair
    display: "Key-Pair Authentication"
    fields:
      - { name: account,                required: true,  secret: false, description: "Snowflake account identifier" }
      - { name: user,                   required: true,  secret: false, description: "Snowflake username" }
      - { name: private_key,            required: true,  secret: true,  description: "PEM-formatted private key content" }
      - { name: private_key_passphrase, required: false, secret: true,  description: "passphrase for encrypted private key" }
      - { name: database,               required: true,  secret: false, description: "database name" }
      - { name: schema,                 required: false, secret: false, description: "schema name" }
      - { name: warehouse,              required: false, secret: false, description: "warehouse to use" }
      - { name: role,                   required: false, secret: false, description: "role to assume" }
test_snippet: |
  import snowflake.connector, os
  conn = snowflake.connector.connect(
      account=os.environ['DS_ACCOUNT'],
      user=os.environ['DS_USER'],
      password=os.environ.get('DS_PASSWORD', ''),
      database=os.environ.get('DS_DATABASE', ''),
      schema=os.environ.get('DS_SCHEMA', 'PUBLIC'),
      warehouse=os.environ.get('DS_WAREHOUSE', ''),
  )
  conn.close()
  print("ok")
```

Account identifier: Admin → Accounts → hover your account name to reveal the identifier.
Format is either `<orgname>-<accountname>` or `<accountlocator>.<region>.<cloud>`.

---

## Google BigQuery

```yaml
engine: bigquery
display_name: Google BigQuery
pip: google-cloud-bigquery
popular: true
name_from: [project_id, dataset]
fields:
  - { name: project_id,           required: true,  secret: false, description: "GCP project ID containing your BigQuery datasets" }
  - { name: dataset,              required: true,  secret: false, description: "BigQuery dataset name" }
  - { name: service_account_json, required: false, secret: true,  description: "contents of service account JSON key file (paste the full JSON)" }
  - { name: service_account_keys, required: false, secret: false, description: "path to service account JSON key file on disk" }
test_snippet: |
  import json, os
  from google.cloud import bigquery
  from google.oauth2 import service_account

  sa_json = os.environ.get('DS_SERVICE_ACCOUNT_JSON', '')
  if sa_json:
      creds = service_account.Credentials.from_service_account_info(
          json.loads(sa_json),
          scopes=['https://www.googleapis.com/auth/bigquery.readonly'],
      )
      client = bigquery.Client(project=os.environ['DS_PROJECT_ID'], credentials=creds)
  else:
      client = bigquery.Client(project=os.environ['DS_PROJECT_ID'])
  list(client.list_datasets())
  print("ok")
```

To create a service account key: GCP Console → IAM → Service Accounts → your account →
Keys → Add Key → JSON. Grant the account `BigQuery Data Viewer` + `BigQuery Job User` roles.

---

## Microsoft SQL Server

```yaml
engine: mssql
display_name: Microsoft SQL Server
pip: pymssql
popular: true
name_from: database
fields:
  - { name: host,     required: true,  secret: false, description: "hostname or IP of the SQL Server (for Azure use server field instead)" }
  - { name: port,     required: true, secret: false, description: "port number", default: "1433" }
  - { name: database, required: true,  secret: false, description: "database name" }
  - { name: user,     required: true,  secret: false, description: "SQL Server username" }
  - { name: password, required: true,  secret: true,  description: "SQL Server password" }
  - { name: server,   required: false, secret: false, description: "server name — use for named instances or Azure SQL (e.g. myserver.database.windows.net)" }
  - { name: schema,   required: false, secret: false, description: "schema name (defaults to dbo)" }
test_snippet: |
  import pymssql, os
  conn = pymssql.connect(
      server=os.environ.get('DS_SERVER') or os.environ['DS_HOST'],
      port=os.environ.get('DS_PORT', '1433'),
      database=os.environ['DS_DATABASE'],
      user=os.environ['DS_USER'],
      password=os.environ['DS_PASSWORD'],
  )
  conn.close()
  print("ok")
```

For Azure SQL Database use `server` field with value like `myserver.database.windows.net`.
For Windows Authentication omit user/password and ensure pymssql is built with Kerberos support.

---

## Amazon Redshift

```yaml
engine: redshift
display_name: Amazon Redshift
pip: psycopg2-binary
popular: true
name_from: [host, database]
fields:
  - { name: host,     required: true,  secret: false, description: "Redshift cluster endpoint (e.g. mycluster.abc123.us-east-1.redshift.amazonaws.com)" }
  - { name: port,     required: true, secret: false, description: "port number", default: "5439" }
  - { name: database, required: true,  secret: false, description: "database name" }
  - { name: user,     required: true,  secret: false, description: "Redshift username" }
  - { name: password, required: true,  secret: true,  description: "Redshift password" }
  - { name: schema,   required: false, secret: false, description: "schema name (defaults to public)" }
  - { name: sslmode,  required: false, secret: false, description: "SSL mode: require (default), verify-ca, verify-full, disable", default: "require" }
test_snippet: |
  import psycopg2, os
  conn = psycopg2.connect(
      host=os.environ['DS_HOST'],
      port=int(os.environ.get('DS_PORT', '5439')),
      dbname=os.environ['DS_DATABASE'],
      user=os.environ['DS_USER'],
      password=os.environ['DS_PASSWORD'],
      sslmode=os.environ.get('DS_SSLMODE', 'require'),
  )
  conn.close()
  print("ok")
```

Redshift is PostgreSQL-compatible. Find the cluster endpoint in AWS Console →
Redshift → Clusters → your cluster → Endpoint (omit the port suffix).

---

## Databricks

```yaml
engine: databricks
display_name: Databricks
pip: databricks-sql-connector
popular: true
name_from: [server_hostname, catalog]
fields:
  - { name: server_hostname,      required: true,  secret: false, description: "server hostname for the cluster or SQL warehouse (from JDBC/ODBC connection string)" }
  - { name: http_path,            required: true,  secret: false, description: "HTTP path of the cluster or SQL warehouse" }
  - { name: access_token,         required: true,  secret: true,  description: "Databricks personal access token" }
  - { name: catalog,              required: false, secret: false, description: "Unity Catalog name (defaults to hive_metastore)" }
  - { name: schema,               required: false, secret: false, description: "schema (database) to use" }
  - { name: session_configuration, required: false, secret: false, description: "Spark session configuration as key=value pairs" }
test_snippet: |
  from databricks import sql as dbsql
  import os
  conn = dbsql.connect(
      server_hostname=os.environ['DS_SERVER_HOSTNAME'],
      http_path=os.environ['DS_HTTP_PATH'],
      access_token=os.environ['DS_ACCESS_TOKEN'],
      catalog=os.environ.get('DS_CATALOG', ''),
      schema=os.environ.get('DS_SCHEMA', ''),
  )
  conn.close()
  print("ok")
```

Personal access token: User Settings → Developer → Access Tokens → Generate New Token.
HTTP path and server hostname: SQL Warehouses → your warehouse → Connection Details tab.

---

## MariaDB

```yaml
engine: mariadb
display_name: MariaDB
pip: mysql-connector-python
popular: true
name_from: database
fields:
  - { name: host,     required: true,  secret: false, description: "hostname or IP of your MariaDB server" }
  - { name: port,     required: true, secret: false, description: "port number", default: "3306" }
  - { name: database, required: true,  secret: false, description: "database name to connect" }
  - { name: user,     required: true,  secret: false, description: "MariaDB username" }
  - { name: password, required: true,  secret: true,  description: "MariaDB password" }
  - { name: ssl,      required: false, secret: false, description: "enable SSL (true/false)" }
  - { name: ssl_ca,   required: false, secret: false, description: "path to CA certificate file" }
  - { name: ssl_cert, required: false, secret: false, description: "path to client certificate file" }
  - { name: ssl_key,  required: false, secret: false, description: "path to client private key file" }
test_snippet: |
  import mysql.connector, os
  conn = mysql.connector.connect(
      host=os.environ['DS_HOST'],
      port=int(os.environ.get('DS_PORT', '3306')),
      database=os.environ['DS_DATABASE'],
      user=os.environ['DS_USER'],
      password=os.environ['DS_PASSWORD'],
  )
  conn.close()
  print("ok")
```

MariaDB is wire-compatible with MySQL, so the mysql-connector-python driver works for both.

---

## HubSpot

```yaml
engine: hubspot
display_name: HubSpot
pip: hubspot-api-client
popular: true
auth_method: choice
auth_methods:
  - name: pat
    display: "Private App Token (recommended)"
    fields:
      - { name: access_token, required: true, secret: true, description: "HubSpot Private App token (starts with pat-na1-)" }
  - name: oauth2
    display: "OAuth2 (for multi-account or publishable apps)"
    fields:
      - { name: client_id,     required: true,  secret: false, description: "OAuth2 client ID" }
      - { name: client_secret, required: true,  secret: true,  description: "OAuth2 client secret" }
    oauth2:
      auth_url: https://app.hubspot.com/oauth/authorize
      token_url: https://api.hubapi.com/oauth/v1/token
      scopes: [crm.objects.contacts.read, crm.objects.deals.read]
      store_fields: [access_token, refresh_token]
test_snippet: |
  import hubspot, os
  client = hubspot.Client.create(access_token=os.environ['DS_ACCESS_TOKEN'])
  client.crm.contacts.basic_api.get_page(limit=1)
  print("ok")
```

For Private App Token: HubSpot → Settings → Integrations → Private Apps → Create.
Recommended scopes: `crm.objects.contacts.read`, `crm.objects.deals.read`, `crm.objects.companies.read`.

For OAuth2: collect client_id and client_secret, then use the scratchpad to:

1. Build the authorization URL using `auth_url` + params above
2. Start a local HTTP server on port 8099 to catch the callback
3. Open the URL in the user's browser with `webbrowser.open()`
4. Extract the `code` from the callback, POST to `token_url` for tokens
5. Return `access_token` and `refresh_token` to store in wallet

---

## Oracle Database

```yaml
engine: oracle_database
display_name: Oracle Database
pip: oracledb
popular: true
name_from: [host, service_name]
fields:
  - { name: user,         required: true,  secret: false, description: "Oracle database username" }
  - { name: password,     required: true,  secret: true,  description: "Oracle database password" }
  - { name: host,         required: true, secret: false, description: "hostname or IP address of the Oracle server" }
  - { name: port,         required: true, secret: false, description: "port number (default 1521)", default: "1521" }
  - { name: service_name, required: false, secret: false, description: "Oracle service name (preferred over SID)" }
  - { name: sid,          required: false, secret: false, description: "Oracle SID — use service_name if possible" }
  - { name: dsn,          required: false, secret: false, description: "full DSN string — overrides host/port/service_name" }
  - { name: auth_mode,    required: false, secret: false, description: "authorization mode (e.g. SYSDBA)" }
test_snippet: |
  import oracledb, os
  dsn = os.environ.get('DS_DSN') or oracledb.makedsn(
      os.environ.get('DS_HOST', 'localhost'),
      os.environ.get('DS_PORT', '1521'),
      service_name=os.environ.get('DS_SERVICE_NAME', ''),
  )
  conn = oracledb.connect(user=os.environ['DS_USER'], password=os.environ['DS_PASSWORD'], dsn=dsn)
  conn.close()
  print("ok")
```

oracledb runs in thin mode by default (no Oracle Client libraries needed).
Set `auth_mode` to `SYSDBA` or `SYSOPER` for privileged connections.

---

## DuckDB

```yaml
engine: duckdb
display_name: DuckDB
pip: duckdb
popular: false
name_from: database
fields:
  - { name: database,         required: false, secret: false, description: "path to DuckDB database file; omit or use :memory: for in-memory database", default: ":memory:" }
  - { name: motherduck_token, required: false, secret: true,  description: "MotherDuck access token for connecting to a MotherDuck cloud database" }
  - { name: read_only,        required: false, secret: false, description: "open in read-only mode (true/false)", default: "false" }
test_snippet: |
  import duckdb, os
  db_path = os.environ.get('DS_DATABASE', ':memory:')
  token = os.environ.get('DS_MOTHERDUCK_TOKEN', '')
  if token:
      conn = duckdb.connect(f'md:{db_path}?motherduck_token={token}')
  else:
      conn = duckdb.connect(db_path)
  conn.execute('SELECT 1').fetchone()
  conn.close()
  print("ok")
```

For MotherDuck, use `database` as the MotherDuck database name (e.g. `my_db`) and provide
the access token. For local files, provide the path to a `.duckdb` file.

---

## pgvector

```yaml
engine: pgvector
display_name: pgvector
pip: pgvector psycopg2-binary
popular: false
name_from: database
fields:
  - { name: host,     required: true,  secret: false, description: "hostname or IP of your PostgreSQL server with pgvector extension" }
  - { name: port,     required: true, secret: false, description: "port number", default: "5432" }
  - { name: database, required: true,  secret: false, description: "database name" }
  - { name: user,     required: true,  secret: false, description: "database username" }
  - { name: password, required: true,  secret: true,  description: "database password" }
  - { name: schema,   required: false, secret: false, description: "schema name (defaults to public)" }
  - { name: sslmode,  required: false, secret: false, description: "SSL mode: prefer, require, disable" }
test_snippet: |
  import psycopg2, os
  conn = psycopg2.connect(
      host=os.environ['DS_HOST'],
      port=int(os.environ.get('DS_PORT', '5432')),
      dbname=os.environ['DS_DATABASE'],
      user=os.environ['DS_USER'],
      password=os.environ['DS_PASSWORD'],
  )
  cur = conn.cursor()
  cur.execute("SELECT extname FROM pg_extension WHERE extname = 'vector'")
  if not cur.fetchone():
      raise RuntimeError("pgvector extension not installed — run: CREATE EXTENSION vector;")
  conn.close()
  print("ok")
```

pgvector must be installed in the PostgreSQL instance: `CREATE EXTENSION IF NOT EXISTS vector;`
Managed options: Supabase, Neon, and AWS RDS for PostgreSQL all support pgvector.

---

## ChromaDB

```yaml
engine: chromadb
display_name: ChromaDB
pip: chromadb
popular: false
name_from: host
fields:
  - { name: host,              required: true, secret: false, description: "ChromaDB server host for HTTP client mode (omit for local in-process mode)" }
  - { name: port,              required: true, secret: false, description: "ChromaDB server port", default: "8000" }
  - { name: persist_directory, required: false, secret: false, description: "local directory for persistent storage (local mode only)" }
test_snippet: |
  import chromadb, os
  host = os.environ.get('DS_HOST', '')
  if host:
      client = chromadb.HttpClient(
          host=host,
          port=int(os.environ.get('DS_PORT', '8000')),
      )
  else:
      persist_dir = os.environ.get('DS_PERSIST_DIRECTORY', '')
      if persist_dir:
          client = chromadb.PersistentClient(path=persist_dir)
      else:
          client = chromadb.EphemeralClient()
  client.heartbeat()
  print("ok")
```

Three modes: HTTP client (connect to a running ChromaDB server), persistent local (file-backed),
or ephemeral in-memory. For production, run `chroma run` to start the HTTP server.

---

## Salesforce

```yaml
engine: salesforce
display_name: Salesforce
pip: salesforce_api
popular: true
name_from: username
fields:
  - { name: username,      required: true,  secret: false, description: "Salesforce account username (email)" }
  - { name: password,      required: true,  secret: true,  description: "Salesforce account password" }
  - { name: client_id,     required: true,  secret: false, description: "consumer key from the connected app" }
  - { name: client_secret, required: true,  secret: true,  description: "consumer secret from the connected app" }
  - { name: is_sandbox,    required: false, secret: false, description: "true to connect to sandbox, false for production", default: "false" }
test_snippet: |
  import salesforce_api, os
  sf = salesforce_api.Salesforce(
      username=os.environ['DS_USERNAME'],
      password=os.environ['DS_PASSWORD'],
      client_id=os.environ['DS_CLIENT_ID'],
      client_secret=os.environ['DS_CLIENT_SECRET'],
      is_sandbox=os.environ.get('DS_IS_SANDBOX', 'false').lower() == 'true',
  )
  sf.query('SELECT Id FROM Account LIMIT 1')
  print("ok")
```

To get client_id and client_secret: Setup → Apps → App Manager → New Connected App.
Enable OAuth, add callback URL, select scopes (api, refresh_token).

---

## Shopify

```yaml
engine: shopify
display_name: Shopify
pip: ShopifyAPI
popular: true
name_from: shop_url
fields:
  - { name: shop_url,      required: true, secret: false, description: "your Shopify store URL (e.g. mystore.myshopify.com)" }
  - { name: client_id,     required: true, secret: false, description: "client ID (API key) of the custom app" }
  - { name: client_secret, required: true, secret: true,  description: "client secret (API secret key) of the custom app" }
test_snippet: |
  import shopify, os
  shop_url = os.environ['DS_SHOP_URL'].rstrip('/')
  if not shop_url.startswith('https://'):
      shop_url = f'https://{shop_url}'
  session = shopify.Session(shop_url, '2024-01', os.environ['DS_CLIENT_SECRET'])
  shopify.ShopifyResource.activate_session(session)
  shopify.Shop.current()
  shopify.ShopifyResource.clear_session()
  print("ok")
```

Create a custom app: Shopify Admin → Settings → Apps → Develop apps → Create an app.
Grant required API permissions (read_products, read_orders, etc.) then install the app.

---

## NetSuite

```yaml
engine: netsuite
display_name: NetSuite
pip: requests-oauthlib>=1.3.1
popular: false
name_from: account_id
fields:
  - { name: account_id,     required: true,  secret: false, description: "NetSuite account/realm ID (e.g. 123456_SB1)" }
  - { name: consumer_key,   required: true,  secret: true,  description: "OAuth consumer key for the NetSuite integration" }
  - { name: consumer_secret,required: true,  secret: true,  description: "OAuth consumer secret for the NetSuite integration" }
  - { name: token_id,       required: true,  secret: true,  description: "Token ID generated for the integration role" }
  - { name: token_secret,   required: true,  secret: true,  description: "Token secret generated for the integration role" }
  - { name: rest_domain,    required: false, secret: false, description: "REST domain override (defaults to https://<account_id>.suitetalk.api.netsuite.com)" }
  - { name: record_types,   required: false, secret: false, description: "Comma-separated NetSuite record types to expose (e.g. customer,item,salesOrder)" }
test_snippet: |
  import os
  from requests_oauthlib import OAuth1Session
  account_id = os.environ['DS_ACCOUNT_ID']
  rest_domain = os.environ.get('DS_REST_DOMAIN') or f'https://{account_id.lower().replace("_", "-")}.suitetalk.api.netsuite.com'
  url = f'{rest_domain.rstrip("/")}/services/rest/record/v1/metadata-catalog/'
  session = OAuth1Session(
      client_key=os.environ['DS_CONSUMER_KEY'],
      client_secret=os.environ['DS_CONSUMER_SECRET'],
      resource_owner_key=os.environ['DS_TOKEN_ID'],
      resource_owner_secret=os.environ['DS_TOKEN_SECRET'],
      realm=account_id,
      signature_method='HMAC-SHA256',
  )
  r = session.get(url, headers={'Prefer': 'transient'})
  assert r.status_code < 400, f'HTTP {r.status_code}: {r.text[:200]}'
  print("ok")
```

NetSuite uses OAuth 1.0a Token-Based Authentication (TBA). Create an integration record in
NetSuite (Setup → Integration → Manage Integrations), then generate token credentials via
Setup → Users/Roles → Access Tokens. The account ID can be found in Setup → Company → Company Information.

---

## Big Commerce

```yaml
engine: bigcommerce
display_name: Big Commerce
pip: httpx
popular: false
name_from: store_hash
fields:
  - { name: api_base,     required: true,  secret: false, description: "Base URL of the BigCommerce API (e.g. https://api.bigcommerce.com/stores/0fh0fh0fh0/v3/)" }
  - { name: access_token, required: true,  secret: true,  description: "API token for authenticating with BigCommerce" }
test_snippet: |
  import httpx, os
  api_base = os.environ['DS_API_BASE'].rstrip('/')
  access_token = os.environ['DS_ACCESS_TOKEN']
  headers = {'X-Auth-Token': access_token}
  r = httpx.get(f'{api_base}/catalog/products', headers=headers)
  assert r.status_code < 400, f'HTTP {r.status_code}: {r.text[:200]}'
  print("ok")
```

BigCommerce API tokens can be created in the BigCommerce control panel under Advanced Settings → API Accounts. Choose "Create API Account", then select "V2/V3 API Token" and grant the necessary permissions (e.g. "Products: Read-Only" to access product data).
---

## TimescaleDB

```yaml
engine: timescaledb
display_name: TimescaleDB
pip: psycopg2-binary
name_from: [host, database]
fields:
  - { name: host,     required: true,  secret: false, description: "hostname or IP of the TimescaleDB server" }
  - { name: port,     required: true, secret: false, description: "port number", default: "5432" }
  - { name: database, required: true,  secret: false, description: "database name" }
  - { name: user,     required: true,  secret: false, description: "database username" }
  - { name: password, required: true,  secret: true,  description: "database password" }
  - { name: schema,   required: false, secret: false, description: "schema name (defaults to public)" }
test_snippet: |
  import psycopg2, os
  conn = psycopg2.connect(
      host=os.environ['DS_HOST'],
      port=int(os.environ.get('DS_PORT', '5432')),
      dbname=os.environ['DS_DATABASE'],
      user=os.environ['DS_USER'],
      password=os.environ['DS_PASSWORD'],
  )
  cur = conn.cursor()
  cur.execute("SELECT extname FROM pg_extension WHERE extname = 'timescaledb'")
  if not cur.fetchone():
      raise RuntimeError("timescaledb extension not found — is TimescaleDB installed?")
  conn.close()
  print("ok")
```

TimescaleDB is a PostgreSQL extension for time-series data. Managed options include Timescale Cloud
and self-hosted PostgreSQL with the TimescaleDB extension installed.

---

## Gmail

```yaml
engine: gmail
display_name: Gmail
name_from: email
popular: true
fields:
  - { name: email,         required: true,  secret: false, description: "your Gmail address (e.g. you@gmail.com)" }
  - { name: app_password,  required: true,  secret: true,  description: "16-character app password from myaccount.google.com/apppasswords" }
test_snippet: |
  import imaplib, os
  imap = imaplib.IMAP4_SSL("imap.gmail.com")
  imap.login(os.environ['DS_EMAIL'], os.environ['DS_APP_PASSWORD'])
  imap.logout()
  print("ok")
```

Requires 2-Factor Authentication enabled on your Google account. Then generate an App Password
at myaccount.google.com/apppasswords — select "Mail" as the app. No OAuth setup needed.

---

## Email

```yaml
engine: email
display_name: Email
name_from: email
popular: false
fields:
  - { name: email,       required: true,  secret: false, description: "email address to connect" }
  - { name: password,    required: true,  secret: true,  description: "email account password or app-specific password" }
  - { name: imap_server, required: false, secret: false, description: "IMAP server hostname", default: "imap.gmail.com" }
  - { name: smtp_server, required: false, secret: false, description: "SMTP server hostname", default: "smtp.gmail.com" }
  - { name: smtp_port,   required: false, secret: false, description: "SMTP port", default: "587" }
test_snippet: |
  import imaplib, os
  imap = imaplib.IMAP4_SSL(os.environ.get('DS_IMAP_SERVER', 'imap.gmail.com'))
  imap.login(os.environ['DS_EMAIL'], os.environ['DS_PASSWORD'])
  imap.logout()
  print("ok")
```

For non-Gmail providers, set imap_server and smtp_server accordingly. Use an app-specific
password if your provider requires it.

---

## Adding a new data source

Follow the YAML format above. Add to `~/.anton/datasources.md` (user overrides).
Anton merges user overrides on top of the built-in registry at startup.
