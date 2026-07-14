from __future__ import annotations

from typing import Annotated

import typer
import uvicorn

app = typer.Typer(no_args_is_help=True, help="Quant research platform Web service")


@app.command()
def serve(
    host: Annotated[str, typer.Option()] = "127.0.0.1",
    port: Annotated[int, typer.Option()] = 8765,
    reload: Annotated[bool, typer.Option()] = False,
) -> None:
    """Start the local FastAPI control plane."""
    uvicorn.run("quant_platform.api:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    app()
