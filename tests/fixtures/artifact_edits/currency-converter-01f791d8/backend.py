import argparse
import math
import os
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from mangum import Mangum

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Secrets ===
# This backend uses only a fixed, synthetic rate table.
SECRETS = {}

# === API routes ===
CURRENCIES = ("USD", "EUR", "GBP", "JPY", "CAD", "AUD", "CHF")
RATES = {
    "USD": 1.0,
    "EUR": 0.92,
    "GBP": 0.79,
    "JPY": 151.2,
    "CAD": 1.36,
    "AUD": 1.52,
    "CHF": 0.88,
}


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=400,
        content={"detail": "Invalid amount or unsupported currency"},
    )


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.get("/api/rates")
async def get_rates():
    return {
        "rates": [
            {"currency": currency, "rate": rate}
            for currency, rate in RATES.items()
        ]
    }


@app.get(
    "/api/convert",
    responses={
        400: {
            "description": "Invalid amount or unsupported currency",
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "properties": {"detail": {"type": "string"}},
                        "required": ["detail"],
                        "additionalProperties": False,
                    }
                }
            },
        }
    },
)
async def convert(
    amount: float = Query(...),
    from_currency: Literal["USD", "EUR", "GBP", "JPY", "CAD", "AUD", "CHF"] = Query(
        ..., alias="from"
    ),
    to_currency: Literal["USD", "EUR", "GBP", "JPY", "CAD", "AUD", "CHF"] = Query(
        ..., alias="to"
    ),
):
    if not math.isfinite(amount):
        return JSONResponse(
            status_code=400,
            content={"detail": "Invalid amount or unsupported currency"},
        )

    converted_amount = amount / RATES[from_currency] * RATES[to_currency]
    return {
        "amount": amount,
        "from": from_currency,
        "to": to_currency,
        "converted_amount": converted_amount,
    }


STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")

handler = Mangum(app, lifespan="off")

if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    uvicorn.run(app, host="127.0.0.1", port=args.port)