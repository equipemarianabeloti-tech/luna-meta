from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
from collections import defaultdict
from typing import Any

import httpx
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, Response
from google import genai
from google.genai import types

from prompt_luna import PROMPT_LUNA


load_dotenv()

VERIFY_TOKEN = os.getenv("VERIFY_TOKEN", "")
META_ACCESS_TOKEN = os.getenv("META_ACCESS_TOKEN", "")
META_PHONE_NUMBER_ID = os.getenv("META_PHONE_NUMBER_ID", "")
META_APP_SECRET = os.getenv("META_APP_SECRET", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
GRAPH_API_VERSION = os.getenv("GRAPH_API_VERSION", "v25.0")

MAX_HISTORY_MESSAGES = 10

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("luna-meta")

app = FastAPI(
    title="Luna Meta",
    version="1.0.0",
)

gemini_client = (
    genai.Client(api_key=GEMINI_API_KEY)
    if GEMINI_API_KEY
    else None
)

# Memória simples em RAM.
# Para a primeira versão, evita banco de dados.
conversation_history: dict[str, list[dict[str, str]]] = defaultdict(list)

# Impede respostas duplicadas enquanto o serviço estiver ativo.
processed_message_ids: set[str] = set()


def validate_environment() -> list[str]:
    required = {
        "VERIFY_TOKEN": VERIFY_TOKEN,
        "META_ACCESS_TOKEN": META_ACCESS_TOKEN,
        "META_PHONE_NUMBER_ID": META_PHONE_NUMBER_ID,
        "GEMINI_API_KEY": GEMINI_API_KEY,
    }

    return [
        name
        for name, value in required.items()
        if not value
    ]


@app.on_event("startup")
async def startup_event() -> None:
    missing = validate_environment()

    if missing:
        logger.warning(
            "Variáveis ausentes: %s",
            ", ".join(missing),
        )
    else:
        logger.info("Luna Meta iniciada corretamente.")


@app.get("/")
async def home() -> dict[str, Any]:
    missing = validate_environment()

    return {
        "status": "online",
        "assistente": "Luna",
        "versao": "1.0.0",
        "configuracao": (
            "completa"
            if not missing
            else "incompleta"
        ),
        "variaveis_ausentes": missing,
    }


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/webhook")
async def verify_webhook(request: Request) -> Response:
    mode = request.query_params.get("hub.mode")
    token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge")

    if (
        mode == "subscribe"
        and token == VERIFY_TOKEN
        and challenge
    ):
        logger.info("Webhook verificado pela Meta.")
        return Response(
            content=challenge,
            media_type="text/plain",
            status_code=200,
        )

    logger.warning("Tentativa inválida de verificação do webhook.")

    raise HTTPException(
        status_code=403,
        detail="Falha na verificação do webhook.",
    )


def validate_meta_signature(
    raw_body: bytes,
    signature_header: str | None,
) -> bool:
    if not META_APP_SECRET:
        # A validação será ativada assim que META_APP_SECRET for preenchida.
        return True

    if not signature_header:
        return False

    if not signature_header.startswith("sha256="):
        return False

    received_signature = signature_header.removeprefix("sha256=")

    expected_signature = hmac.new(
        META_APP_SECRET.encode("utf-8"),
        raw_body,
        hashlib.sha256,
    ).hexdigest()

    return hmac.compare_digest(
        received_signature,
        expected_signature,
    )


@app.post("/webhook")
async def receive_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
) -> dict[str, str]:
    raw_body = await request.body()

    signature = request.headers.get(
        "x-hub-signature-256"
    )

    if not validate_meta_signature(
        raw_body,
        signature,
    ):
        logger.warning("Assinatura inválida recebida da Meta.")

        raise HTTPException(
            status_code=401,
            detail="Assinatura inválida.",
        )

    try:
        payload = json.loads(raw_body)
    except json.JSONDecodeError as error:
        logger.error("JSON inválido: %s", error)

        raise HTTPException(
            status_code=400,
            detail="JSON inválido.",
        ) from error

    background_tasks.add_task(
        process_webhook_payload,
        payload,
    )

    # A Meta precisa receber confirmação rapidamente.
    return {"status": "received"}


async def process_webhook_payload(
    payload: dict[str, Any],
) -> None:
    try:
        entries = payload.get("entry", [])

        for entry in entries:
            changes = entry.get("changes", [])

            for change in changes:
                if change.get("field") != "messages":
                    continue

                value = change.get("value", {})
                messages = value.get("messages", [])

                for message in messages:
                    await process_message(message)

    except Exception:
        logger.exception(
            "Erro inesperado ao processar webhook."
        )


async def process_message(
    message: dict[str, Any],
) -> None:
    message_id = message.get("id")
    sender = message.get("from")
    message_type = message.get("type")

    if not message_id or not sender:
        return

    if message_id in processed_message_ids:
        logger.info(
            "Mensagem duplicada ignorada: %s",
            message_id,
        )
        return

    processed_message_ids.add(message_id)

    if len(processed_message_ids) > 5000:
        processed_message_ids.clear()

    if message_type != "text":
        await send_whatsapp_message(
            sender,
            (
                "Por enquanto consigo atender melhor por texto. "
                "Pode escrever sua mensagem para mim? 🌸"
            ),
        )
        return

    user_text = (
        message
        .get("text", {})
        .get("body", "")
        .strip()
    )

    if not user_text:
        return

    logger.info(
        "Mensagem recebida de %s: %s",
        sender,
        user_text,
    )

    try:
        answer = await generate_luna_response(
            sender,
            user_text,
        )

        await send_whatsapp_message(
            sender,
            answer,
        )

    except Exception:
        logger.exception(
            "Falha ao responder a mensagem %s.",
            message_id,
        )

        await send_whatsapp_message(
            sender,
            (
                "Tive uma pequena dificuldade por aqui. "
                "Pode me enviar sua mensagem novamente em alguns instantes? 💛"
            ),
        )


async def generate_luna_response(
    phone_number: str,
    user_text: str,
) -> str:
    if gemini_client is None:
        raise RuntimeError(
            "GEMINI_API_KEY não configurada."
        )

    history = conversation_history[phone_number]

    history_text = "\n".join(
        (
            f"{item['role']}: {item['text']}"
            for item in history[-MAX_HISTORY_MESSAGES:]
        )
    )

    prompt = f"""
{PROMPT_LUNA}

HISTÓRICO RECENTE DA CONVERSA:
{history_text if history_text else "Ainda não há histórico."}

NOVA MENSAGEM DA CLIENTE:
{user_text}

Escreva somente a resposta que será enviada pelo WhatsApp.
Não inclua explicações, títulos, aspas ou observações internas.
"""

    def call_gemini() -> str:
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.7,
                max_output_tokens=350,
            ),
        )

        return (response.text or "").strip()

    answer = await asyncio.to_thread(call_gemini)

    if not answer:
        answer = (
            "Estou por aqui para te ajudar. "
            "Pode me contar um pouco mais? 🌸"
        )

    history.append(
        {
            "role": "Cliente",
            "text": user_text,
        }
    )

    history.append(
        {
            "role": "Luna",
            "text": answer,
        }
    )

    conversation_history[phone_number] = history[
        -MAX_HISTORY_MESSAGES:
    ]

    return answer


async def send_whatsapp_message(
    recipient: str,
    text: str,
) -> None:
    if not META_ACCESS_TOKEN:
        raise RuntimeError(
            "META_ACCESS_TOKEN não configurado."
        )

    if not META_PHONE_NUMBER_ID:
        raise RuntimeError(
            "META_PHONE_NUMBER_ID não configurado."
        )

    url = (
        f"https://graph.facebook.com/"
        f"{GRAPH_API_VERSION}/"
        f"{META_PHONE_NUMBER_ID}/messages"
    )

    headers = {
        "Authorization": (
            f"Bearer {META_ACCESS_TOKEN}"
        ),
        "Content-Type": "application/json",
    }

    payload = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": recipient,
        "type": "text",
        "text": {
            "preview_url": False,
            "body": text[:4096],
        },
    }

    async with httpx.AsyncClient(
        timeout=30.0
    ) as client:
        response = await client.post(
            url,
            headers=headers,
            json=payload,
        )

    if response.status_code >= 400:
        logger.error(
            "Erro da Meta: status=%s resposta=%s",
            response.status_code,
            response.text,
        )

        response.raise_for_status()

    logger.info(
        "Resposta enviada para %s.",
        recipient,
    )
