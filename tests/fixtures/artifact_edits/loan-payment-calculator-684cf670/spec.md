## Insights
- Monthly payment and total interest: summarize the default loan’s monthly cost and borrowing cost; default values are **$1,580.17 per month** and **$318,861.22 total interest**.
- Yearly amortization table: show how annual principal and interest payments reduce the balance; for the defaults, year 1 is **$2,794.31 principal**, **$16,167.73 interest**, **$247,205.69 remaining**, and year 30 is **$18,310.96 principal**, **$651.08 interest**, **$0.00 remaining**.

## Implementation notes
- Keep edited input values separate from the last calculated results; update results only when **Calculate** is pressed. Initialize the page with results calculated from the defaults.
- Use the standard fixed-payment amortization calculation with monthly rate = annual rate ÷ 12 and number of payments = years × 12. For a 0% rate, monthly payment is loan amount ÷ number of payments.
- Aggregate each set of 12 payments into one table row; display monetary values in USD rounded to cents. Use the unrounded payment and balances for calculations, rounding only displayed values. Clamp the final remaining balance to $0.00.
- Validate that loan amount and term are positive and the annual rate is nonnegative; do not replace the displayed results until a valid calculation is submitted.