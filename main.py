from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import user_router, auth_router, post_router, comment_router
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request                     