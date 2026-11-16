# Currency Converter PRD

## Goal
A currency converter that calculates an amount between supported currencies using fixed synthetic exchange rates and displays the full rate table.

## Artifact type
**Fullstack-stateless-app** — a Python backend serves rates and conversions to a static frontend.

## Data model
- **Fixed exchange-rate table:** currency code and rate in currency units per 1 USD. Sample rows: USD — 1.0; EUR — 0.92; GBP — 0.79.

## Functional requirements
1. Provide a small Python backend and a static frontend.
2. Support USD at a rate of 1.0, along with EUR 0.92, GBP 0.79, JPY 151.2, CAD 1.36, AUD 1.52, and CHF 0.88.
3. Let users enter an amount, choose source and destination currencies, and request a conversion.
4. Show the converted amount.
5. Display a table of all supported currency rates.

## UI/UX requirements
- Include an amount input, source and destination currency selectors, a Convert button, and an area for the result.
- Include a table listing all supported currency rates.