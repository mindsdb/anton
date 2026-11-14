## Insights
- Conversion result: communicates the calculated value for the selected amount and currency pair.
- Rates table: shows the relative value of each supported currency against USD for reference.

## Implementation notes
- Keep the fixed rates in the backend; treat them as units of a currency per 1 USD, with USD as the 1.0 baseline.
- Convert using `amount / rate_from * rate_to`.
- Populate both currency selectors from the rates endpoint after loading rates.
- Show a clear error if rates cannot be loaded or the conversion cannot be completed.
- Button label: “Convert”.

## Backend
- GET /api/rates: provides the fixed synthetic exchange rates.
- GET /api/convert: calculates a conversion between supported currencies.
- The frontend loads rates from the backend for the selectors and table, then requests conversions from the backend. The backend uses its fixed rate table and does not read from an external source.