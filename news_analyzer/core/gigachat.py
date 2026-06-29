import os
import re
import json
import time
import base64
import requests
from io import BytesIO

class GigaChatClient:
    _instance = None

    def __init__(self):
        self.api_url = os.getenv('GIGACHAT_API_URL', '').rstrip('/')
        self.token = os.getenv('JPY_API_TOKEN', '')
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }
        self.text_model = "GigaChat-3-Ultra"
        self.image_model = "GigaChat-3-Ultra"
        self.embedding_model = os.getenv("GIGACHAT_EMBEDDING_MODEL", "EmbeddingsGigaR")
        print(f"GigaChatClient initialized: {self.api_url}, text_model={self.text_model}, image_model={self.image_model}, embedding_model={self.embedding_model}")

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            print("Creating GigaChatClient instance...")
            cls._instance = cls()
        return cls._instance

    def list_models(self):
        """Список доступных моделей из GigaChat /models (id-строки)."""
        response = requests.get(url=self.api_url + "/models", headers=self.headers, timeout=30)
        if not response.ok:
            raise Exception(f"GigaChat /models error {response.status_code}: {response.text}")
        data = response.json().get("data", [])
        return [item["id"] for item in data if "id" in item]

    def completions(self, messages, model=None, temperature=0, n=1):
        data = {
            "model": model or self.text_model,
            "messages": messages,
            "n": n,
            "temperature": temperature,
            "top_p": 0
        }
        response = requests.post(
            url=self.api_url + "/chat/completions",
            headers=self.headers,
            json=data,
            timeout=120,
        )
        if response.ok:
            return response.json()
        else:
            raise Exception(f"GigaChat API error {response.status_code}: {response.text}")

    def rate_importance(self, text, prompt, model=None):
        messages = [
            {
                "role": "system",
                "content": prompt
            },
            {
                "role": "user",
                "content": text[:30000]
            }
        ]
        resp = self.completions(messages, temperature=0, model=model)
        response_text = resp["choices"][0]["message"]["content"].strip()
        
        try:
            clean_text = re.sub(r'^```json\s*', '', response_text)
            clean_text = re.sub(r'\s*```$', '', clean_text).strip()
            
            data = json.loads(clean_text)
            score = float(data.get("score", 0))
            reasoning = str(data.get("reasoning", ""))
            return score, reasoning
            
        except (json.JSONDecodeError, ValueError, TypeError):
            match = re.search(r'(\d+(?:\.\d+)?)', response_text)
            if match:
                return float(match.group(1)), f"Оригинальный ответ (JSON Parse Error): {response_text[:200]}"
            return 0, f"ERROR. Не удалось определить оценку. Оригинальный ответ: {response_text[:200]}"

    def embeddings(self, texts, model=None):
        """Вернуть список векторов (list[list[float]]) для входных текстов.

        Принимает строку или список строк. Использует endpoint /embeddings
        GigaChat. Порядок векторов соответствует порядку входных текстов.
        """
        if isinstance(texts, str):
            texts = [texts]
        payload = {
            "model": model or self.embedding_model,
            "input": [t[:4000] for t in texts],
        }
        response = requests.post(
            url=self.api_url + "/embeddings",
            headers=self.headers,
            json=payload,
            timeout=120,
        )
        if not response.ok:
            raise Exception(f"GigaChat embeddings error {response.status_code}: {response.text}")
        data = response.json()["data"]
        # сортируем по index на случай, если API вернёт не по порядку
        data = sorted(data, key=lambda item: item.get("index", 0))
        return [item["embedding"] for item in data]

    def summarize_news(self, text, prompt, model=None):
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Текст новости:\n{text[:8000]}"}
        ]
        resp = self.completions(messages, temperature=0.3, model=model)
        return resp["choices"][0]["message"]["content"].strip()

    def generate_image(self, summary_text, prompt, model=None):
        payload = {
            "model": model or self.image_model,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Нарисуй иллюстрацию в стиле Flat Design для следующей новости: {summary_text[:2000]}"}
            ],
            "function_call": {
                "name": "text2image"
            }
        }
        response = requests.post(
            self.api_url + "/chat/completions",
            json=payload,
            headers=self.headers,
            timeout=180,
        )
        if not response.ok:
            raise Exception(f"GigaChat image API error {response.status_code}: {response.text}")

        resp_json = response.json()
        content = resp_json["choices"][0]["message"]["content"]

        match = re.search(r'<img\s+src="([^"]+)"', content)
        if not match:
            match = re.search(r"<img\s+src='([^']+)'", content)
        if not match:
            raise Exception(f"Cannot extract file_id from image response: {content[:200]}")

        file_id = match.group(1)

        download_url = f"{self.api_url}/files/{file_id}/content"
        img_response = requests.get(download_url, headers=self.headers, timeout=60)

        if not img_response.ok:
            raise Exception(f"Image download error {img_response.status_code}")

        img_bytes = img_response.content
        b64 = base64.b64encode(img_bytes).decode('utf-8')
        return f"data:image/jpeg;base64,{b64}"