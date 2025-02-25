import asyncio
import json
import time
import httpx
import pandas as pd
import matplotlib.pyplot as plt
from prompts import SAMPLE_PROMPTS

# Конфигурация теста
BALANCER_URL = "http://localhost:1105/generate"
NUM_REQUESTS = 25  # Общее количество запросов
CONCURRENT_REQUESTS = 3  # Максимальное число одновременных запросов
TEST_DURATION_SECONDS = 300  # Продолжительность теста


# Функция для выполнения запроса и измерения времени
async def send_request(prompt, client, request_id):
    start_time = time.time()
    request_data = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 256, "temperature": 0.7, "top_p": 0.7},
    }

    try:
        response = await client.post(BALANCER_URL, json=request_data)
        response_text = response.json()
        status_code = response.status_code
    except Exception as e:
        end_time = time.time()
        return {
            "request_id": request_id,
            "prompt_length": len(prompt),
            "status": "error",
            "error": str(e),
            "response_time": end_time - start_time,
        }

    end_time = time.time()

    try:
        response_data = response.json()
        output_length = len(response_data.get("generated_text", ""))
    except:
        output_length = 0

    return {
        "request_id": request_id,
        "prompt_length": len(prompt),
        "output_length": output_length,
        "status": "success" if status_code == 200 else f"error-{status_code}",
        "response_time": end_time - start_time,
        "request_text": prompt,
        "response_text": response_text.get("generated_text"),
    }


# Основная функция нагрузочного тестирования
async def run_load_test():
    print(
        f"Запуск нагрузочного тестирования: {NUM_REQUESTS} запросов, макс. {CONCURRENT_REQUESTS} одновременно"
    )

    results = []
    start_test_time = time.time()
    limiter = asyncio.Semaphore(CONCURRENT_REQUESTS)

    async with httpx.AsyncClient(timeout=300.0) as client:
        tasks = []

        # Создаем запросы с разной сложностью
        for i in range(NUM_REQUESTS):
            prompt = SAMPLE_PROMPTS[i % len(SAMPLE_PROMPTS)]

            async def bounded_request(req_id, prompt_text):
                async with limiter:
                    return await send_request(prompt_text, client, req_id)

            tasks.append(bounded_request(i, prompt))

        # Запускаем тесты и собираем результаты
        for future in asyncio.as_completed(tasks):
            result = await future
            results.append(result)

            elapsed = time.time() - start_test_time
            rps = len(results) / elapsed
            print(
                f"Запрос {result['request_id']} завершен за {result['response_time']:.2f}с ({len(results)}/{NUM_REQUESTS}, {rps:.2f} RPS)"
            )

            # Прерываем тест, если превышено время
            if elapsed > TEST_DURATION_SECONDS:
                print(f"Превышено время тестирования ({TEST_DURATION_SECONDS}с)")
                break

    end_test_time = time.time()
    total_time = end_test_time - start_test_time

    print(f"\nТестирование завершено за {total_time:.2f} секунд")
    print(f"Выполнено {len(results)}/{NUM_REQUESTS} запросов")
    print(f"Средний RPS: {len(results) / total_time:.2f}")

    # Анализ результатов
    analyze_results(results, total_time)

    # Сохранение ответов
    save_answers(results)


# Анализ и визуализация результатов
def analyze_results(results, total_time):
    df = pd.DataFrame(results)

    # Базовая статистика
    success_count = len(df[df["status"] == "success"])
    error_count = len(df[df["status"] != "success"])
    success_rate = success_count / len(df) * 100

    avg_response_time = df["response_time"].mean()
    p95_response_time = df["response_time"].quantile(0.95)

    print(f"\nРезультаты тестирования:")
    print(f"Успешных запросов: {success_count} ({success_rate:.1f}%)")
    print(f"Ошибок: {error_count}")
    print(f"Среднее время ответа: {avg_response_time:.2f}с")
    print(f"95-й перцентиль времени ответа: {p95_response_time:.2f}с")
    print(f"Общий RPS: {len(results) / total_time:.2f}")

    # Создаем визуализации
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))

    # График времени ответа
    df["request_id"] = df["request_id"].astype(int)
    df = df.sort_values("request_id")

    ax1.plot(df["request_id"], df["response_time"])
    ax1.set_title("Время ответа по запросам")
    ax1.set_xlabel("ID запроса")
    ax1.set_ylabel("Время ответа (с)")
    ax1.grid(True)

    # Гистограмма времени ответа
    ax2.hist(df["response_time"], bins=20, alpha=0.7)
    ax2.axvline(
        avg_response_time,
        color="r",
        linestyle="dashed",
        linewidth=1,
        label=f"Среднее: {avg_response_time:.2f}с",
    )
    ax2.axvline(
        p95_response_time,
        color="g",
        linestyle="dashed",
        linewidth=1,
        label=f"P95: {p95_response_time:.2f}с",
    )
    ax2.set_title("Распределение времени ответа")
    ax2.set_xlabel("Время ответа (с)")
    ax2.set_ylabel("Количество запросов")
    ax2.legend()
    ax2.grid(True)

    # Соотношение длины запроса и времени ответа
    ax3.scatter(df["prompt_length"], df["response_time"])
    ax3.set_title("Зависимость времени ответа от длины запроса")
    ax3.set_xlabel("Длина запроса (символы)")
    ax3.set_ylabel("Время ответа (с)")
    ax3.grid(True)

    plt.tight_layout()
    plt.savefig("load_test_results.png")
    print("График результатов сохранен в файл 'load_test_results.png'")

    # Сохраняем детальные результаты в CSV
    df.to_csv("load_test_detailed_results.csv", index=False)
    print("Детальные результаты сохранены в файл 'load_test_detailed_results.csv'")


def save_answers(results: list[dict]):
    data = {}
    for result in results:
        if result.get("status") == "success":
            data[result.get("request_id")] = {
                "request": result.get("request_text"),
                "response": result.get("response_text"),
            }
    with open("answers.json", "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)
    print("Ответы сохранены в 'answers.json'")


# Запуск теста
if __name__ == "__main__":
    asyncio.run(run_load_test())
