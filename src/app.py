# https://github.com/langchain-ai/langserve
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes

app = FastAPI(
    title="Langchain Langgraph Examples Server",
    version="1.0",
    description="A simple api server using Langchain's Runnable interfaces",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

from _langchain.example001.app import chain as chain001
add_routes(app, chain001, path="/chain001", playground_type="default")

from _langchain.example002.app import chain as chain002
add_routes(app, chain002, path="/chain002", playground_type="default")

from _langchain.example003.app import chain as chain003
add_routes(app, chain003, path="/chain003", playground_type="default") #, config_keys=["configurable"])

from _langchain.example004.app import chain as chain004
add_routes(app, chain004, path="/chain004", playground_type="default") #, config_keys=["configurable"])

from _langchain.example005.app import chain as chain005
add_routes(app, chain005, path="/chain005", playground_type="default")

from _langchain.example006.app import chain as chain006
add_routes(app, chain006, path="/chain006", playground_type="default")

from _langchain.example007.app import chain as chain007
add_routes(app, chain007, path="/chain007", playground_type="default")

from _langchain.example008.app import chain as chain008
add_routes(app, chain008, path="/chain008", playground_type="default")

from _langchain.example009.app import chain as chain009
add_routes(app, chain009, path="/chain009", playground_type="default")

from _langchain.example010.app import chain as chain010
add_routes(app, chain010, path="/chain010", playground_type="default")

from _langchain.example011.app import chain as chain011
add_routes(app, chain011, path="/chain011", playground_type="default")

from _langgraph.example001.app import graph as graph001
add_routes(app, graph001, path="/graph001", playground_type="default")

if __name__ == "__main__":

    uvicorn.run(app, host="localhost", port=8000)
