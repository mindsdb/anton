# Loan Payment Calculator PRD

## Goal
A single-page loan calculator that estimates monthly payments and shows how principal, interest, and remaining balance change each year.

## Artifact type
Single-page HTML app, because it calculates results from user-entered values without relying on external data or saved information.

## Data model
No external data or saved information; calculations use values entered in the calculator.

## Functional requirements
1. Provide inputs for loan amount, annual interest rate, and loan term in years, defaulting to 250,000, 6.5%, and 30 years.
2. When the user presses **Calculate**, show the monthly payment and total interest.
3. Show a yearly amortization table with principal paid, interest paid, and remaining balance for each year.
4. Calculate fixed monthly payments using monthly compounding and group payments into 12-month yearly rows.
5. Display monetary amounts in USD, rounded to cents.

## UI/UX requirements
- Present the loan inputs and **Calculate** button on the page.
- Present the monthly payment and total interest as results.
- Include a yearly amortization table with columns for year, principal paid, interest paid, and remaining balance.