import asyncio
import logging
import time
import uuid
from datetime import datetime
from typing import Dict, Any

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
import random
import json

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("tgi_balancer.log"), logging.StreamHandler()],
)
logger = logging.getLogger("tgi-balancer")


# Модели для запросов и ответов
class GenerationRequest(BaseModel):
    inputs: str
    parameters: Dict[str, Any] = Field(default_factory=dict)  # noqa
    stream: bool = False


class ServerStats(BaseModel):
    url: str
    active_requests: int = 0
    total_requests: int = 0
    success_requests: int = 0
    failed_requests: int = 0
    avg_response_time: float = 0.0
    last_response_time: float = 0.0
    last_active: datetime = Field(default_factory=datetime.now)


# Инициализация FastAPI
app = FastAPI(title="TGI Load Balancer")

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Конфигурация балансировщика
TGI_SERVERS = [
    "http://localhost:1101",  # GPU 0
    "http://localhost:1102",  # GPU 1
    # Добавьте больше серверов по необходимости
]

# Статистика серверов
servers_stats = {url: ServerStats(url=url) for url in TGI_SERVERS}

# HTTP клиент с таймаутом
http_client = httpx.AsyncClient(timeout=300.0)

# Mutex для обновления статистики
stats_lock = asyncio.Lock()


# Функция выбора наименее загруженного сервера
async def select_server() -> str:
    async with stats_lock:
        # Алгоритм Least Connection - выбор сервера с наименьшим количеством активных запросов
        servers = sorted(servers_stats.values(), key=lambda s: s.active_requests)
        selected = servers[0]
        logger.info(
            f"Выбран сервер: {selected.url} (активных запросов: {selected.active_requests})"
        )
        return selected.url


# Обновление статистики сервера при запросе
async def update_stats_request(server_url: str):
    async with stats_lock:
        server = servers_stats[server_url]
        server.active_requests += 1
        server.total_requests += 1
        server.last_active = datetime.now()
        logger.debug(
            f"Сервер {server_url}: +1 активный запрос (всего: {server.active_requests})"
        )


# Обновление статистики сервера при завершении запроса
async def update_stats_response(server_url: str, response_time: float, success: bool):
    async with stats_lock:
        server = servers_stats[server_url]
        server.active_requests -= 1
        server.last_response_time = response_time

        # Обновление средней скорости ответа
        if server.avg_response_time == 0:
            server.avg_response_time = response_time
        else:
            server.avg_response_time = (server.avg_response_time * 0.9) + (
                response_time * 0.1
            )

        if success:
            server.success_requests += 1
        else:
            server.failed_requests += 1

        logger.debug(
            f"Сервер {server_url}: запрос завершен за {response_time:.4f}с, активных: {server.active_requests}"
        )


# Эндпоинт для проверки здоровья балансировщика
@app.get("/health")
async def health_check():
    return {"status": "ok", "servers": len(TGI_SERVERS)}


# Эндпоинт для получения статистики по серверам
@app.get("/stats")
async def get_stats():
    async with stats_lock:
        return {
            "total_servers": len(TGI_SERVERS),
            "total_requests": sum(s.total_requests for s in servers_stats.values()),
            "active_requests": sum(s.active_requests for s in servers_stats.values()),
            "servers": {k: v.model_dump() for k, v in servers_stats.items()},
        }


# Основной эндпоинт для генерации текста
@app.post("/generate")
async def generate(request: GenerationRequest, background_tasks: BackgroundTasks):
    request_id = str(uuid.uuid4())
    start_time = time.time()

    logger.info(
        f"Получен запрос {request_id}: длина входного текста {len(request.inputs)} символов, stream={request.stream}"
    )

    # Выбор сервера
    server_url = await select_server()
    tgi_endpoint = f"{server_url}/generate"

    # Обновление статистики запроса
    await update_stats_request(server_url)

    try:
        if request.stream:
            return await handle_streaming_request(
                request, server_url, request_id, start_time
            )
        else:
            return await handle_normal_request(
                request, server_url, request_id, start_time
            )
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(
            f"Ошибка в запросе {request_id} к {server_url}: {str(e)}, время: {elapsed:.4f}с"
        )

        # Обновление статистики при ошибке
        background_tasks.add_task(update_stats_response, server_url, elapsed, False)

        # Попробуем другой сервер, если доступен
        if len(TGI_SERVERS) > 1:
            fallback_servers = [s for s in TGI_SERVERS if s != server_url]
            fallback_server = random.choice(fallback_servers)

            logger.info(
                f"Повторная попытка запроса {request_id} на резервном сервере {fallback_server}"
            )

            try:
                await update_stats_request(fallback_server)
                fallback_endpoint = f"{fallback_server}/generate"

                if request.stream:
                    return await stream_from_tgi(
                        fallback_endpoint,
                        request,
                        request_id,
                        fallback_server,
                        start_time,
                        background_tasks,
                    )
                else:
                    response = await http_client.post(
                        fallback_endpoint, json=request.model_dump()
                    )

                    new_elapsed = time.time() - start_time
                    logger.info(
                        f"Запрос {request_id} к резервному серверу {fallback_server} выполнен за {new_elapsed:.4f}с"
                    )
                    background_tasks.add_task(
                        update_stats_response, fallback_server, new_elapsed, True
                    )

                    return JSONResponse(content=response.json())
            except Exception as fallback_error:
                new_elapsed = time.time() - start_time
                logger.error(
                    f"Ошибка в резервном запросе {request_id} к {fallback_server}: {str(fallback_error)}, время: {new_elapsed:.4f}с"
                )
                background_tasks.add_task(
                    update_stats_response, fallback_server, new_elapsed, False
                )

                raise HTTPException(
                    status_code=500,
                    detail=f"All TGI servers failed: {str(e)}, Fallback error: {str(fallback_error)}",
                )

        raise HTTPException(status_code=500, detail=str(e))


# Обработка обычного (не потокового) запроса
async def handle_normal_request(
    request: GenerationRequest, server_url: str, request_id: str, start_time: float
):
    tgi_endpoint = f"{server_url}/generate"

    response = await http_client.post(tgi_endpoint, json=request.model_dump())

    elapsed = time.time() - start_time
    logger.info(f"Запрос {request_id} к {server_url} выполнен за {elapsed:.4f}с")

    # Обновление статистики при успешном запросе
    await update_stats_response(server_url, elapsed, True)

    return JSONResponse(content=response.json())


# Обработка потокового запроса
async def handle_streaming_request(
    request: GenerationRequest,
    server_url: str,
    request_id: str,
    start_time: float,
    background_tasks: BackgroundTasks = None,
):
    tgi_endpoint = f"{server_url}/generate"
    return await stream_from_tgi(
        tgi_endpoint, request, request_id, server_url, start_time, background_tasks
    )


# Создание потока от TGI к клиенту
async def stream_from_tgi(
    endpoint: str,
    request: GenerationRequest,
    request_id: str,
    server_url: str,
    start_time: float,
    background_tasks: BackgroundTasks = None,
):
    async def generate_stream():
        chunk_counter = 0
        try:
            async with http_client.stream(
                "POST", endpoint, json=request.model_dump(), timeout=300.0
            ) as response:
                async for chunk in response.aiter_text():
                    chunk_counter += 1
                    yield chunk

            elapsed = time.time() - start_time
            logger.info(
                f"Стриминг-запрос {request_id} к {server_url} выполнен за {elapsed:.4f}с, отправлено {chunk_counter} чанков"
            )

            if background_tasks:
                background_tasks.add_task(
                    update_stats_response, server_url, elapsed, True
                )
            else:
                await update_stats_response(server_url, elapsed, True)

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(
                f"Ошибка в стриминг-запросе {request_id} к {server_url}: {str(e)}, время: {elapsed:.4f}с, отправлено {chunk_counter} чанков"
            )

            if background_tasks:
                background_tasks.add_task(
                    update_stats_response, server_url, elapsed, False
                )
            else:
                await update_stats_response(server_url, elapsed, False)

            # Если был отправлен хотя бы один чанк, завершаем поток вместо ошибки
            if chunk_counter > 0:
                yield json.dumps({"error": str(e)})
            else:
                raise e

    return StreamingResponse(generate_stream(), media_type="application/json")


# Эндпоинт для переадресации на model-info TGI сервера
@app.get("/info")
async def model_info():
    # Всегда берем информацию с первого сервера, так как она должна быть одинакова
    server_url = TGI_SERVERS[0]
    try:
        response = await http_client.get(f"{server_url}/info")
        return JSONResponse(content=response.json())
    except Exception as e:
        logger.error(f"Ошибка при получении информации о модели: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Эндпоинт для проверки здоровья TGI серверов
@app.get("/check-servers")
async def check_servers():
    results = {}
    for server in TGI_SERVERS:
        try:
            start = time.time()
            response = await http_client.get(f"{server}/health")
            elapsed = time.time() - start
            results[server] = {
                "status": "healthy" if response.status_code == 200 else "unhealthy",
                "response_time": elapsed,
                "status_code": response.status_code,
            }
        except Exception as e:
            results[server] = {"status": "error", "error": str(e)}

    return results


# Запуск сервера
if __name__ == "__main__":
    uvicorn.run("tgi_balancer:app", host="0.0.0.0", port=8000, reload=True)
