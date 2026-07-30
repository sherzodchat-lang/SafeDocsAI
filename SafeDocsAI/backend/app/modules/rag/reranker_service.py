"""
Qwen3-Reranker-4B cross-encoder reranker.

Uses AutoModelForCausalLM: feeds (query, document) pairs through the model
and reads the logit ratio of "yes" vs "no" tokens as the relevance score.

Модель зашита в _MODEL_NAME и НЕ берётся из runtime-настройки reranker_model:
промпт (_PREFIX/_SUFFIX) и трюк с логитами yes/no специфичны для Qwen3-Reranker,
на произвольной модели из настроек они дали бы мусорные оценки. Настройка
reranker_model остаётся только выключателем/полем UI; чтобы расхождение не было
молчаливым, о нём предупреждает лог при первом использовании.
"""
import logging
import threading
from typing import Any

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_MODEL_NAME = "Qwen/Qwen3-Reranker-4B"

_PREFIX = (
    "<|im_start|>system\n"
    "Judge whether the Document meets the requirements based on the Query and the "
    "Candidate Document, output your judgement, the answer should be \"yes\" or \"no\"."
    "<|im_end|>\n"
    "<|im_start|>user\n"
)
_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n"

_lock = threading.Lock()
_tokenizer = None
_model = None
_token_yes: int | None = None
_token_no: int | None = None
_device = "cuda"
_warned_about_model = False


def _model_device() -> torch.device:
    if _model is None:
        raise RuntimeError("Reranker model is not loaded")
    return next(_model.parameters()).device


def _load_on_device(target_device: str):
    global _tokenizer, _model, _token_yes, _token_no, _device
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME, padding_side="left")

    logger.info("Loading Qwen3-Reranker-4B onto %s...", target_device.upper())
    load_kwargs = {
        "torch_dtype": torch.float16 if target_device == "cuda" else torch.float32,
        "device_map": target_device,
    }
    _model = AutoModelForCausalLM.from_pretrained(_MODEL_NAME, **load_kwargs)
    _model.eval()
    _token_yes = _tokenizer.convert_tokens_to_ids("yes")
    _token_no = _tokenizer.convert_tokens_to_ids("no")
    _device = target_device
    logger.info(
        "Qwen3-Reranker-4B loaded on %s. yes=%d no=%d",
        target_device.upper(),
        _token_yes,
        _token_no,
    )


def _load_model():
    global _model
    with _lock:
        if _model is not None:
            return
        try:
            _load_on_device("cuda")
        except Exception as exc:
            logger.warning(
                "Qwen3-Reranker GPU load failed, retrying on CPU: %s",
                exc,
            )
            _model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            _load_on_device("cpu")


def _format_input(query: str, document: str) -> str:
    return (
        _PREFIX
        + f"<Query>{query}</Query>\n"
        + f"<Document>{document[:1200]}</Document>\n"
        + _SUFFIX
    )


def _score_batch(queries: list[str], documents: list[str]) -> list[float]:
    global _model
    _load_model()
    inputs_text = [_format_input(q, d) for q, d in zip(queries, documents)]
    inputs = _tokenizer(
        inputs_text,
        padding=True,
        truncation=True,
        max_length=2048,
        return_tensors="pt",
    ).to(_model_device())

    try:
        with torch.no_grad():
            logits = _model(**inputs).logits[:, -1, :]  # last token logits
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower() or _device != "cuda":
            raise
        logger.warning("Qwen3-Reranker GPU inference failed, retrying on CPU: %s", exc)
        with _lock:
            _model = None
            torch.cuda.empty_cache()
            _load_on_device("cpu")
        inputs = inputs.to(_model_device())
        with torch.no_grad():
            logits = _model(**inputs).logits[:, -1, :]

    yes_logits = logits[:, _token_yes]
    no_logits = logits[:, _token_no]
    scores = F.softmax(torch.stack([no_logits, yes_logits], dim=-1), dim=-1)[:, 1]
    return scores.cpu().float().tolist()


def _warn_once_about_ignored_model(model: str) -> None:
    global _warned_about_model
    configured = (model or "").strip()
    if not configured or _warned_about_model or configured == _MODEL_NAME:
        return
    _warned_about_model = True
    logger.warning(
        "Runtime setting reranker_model=%r is ignored: the reranker is hardwired to %s.",
        configured,
        _MODEL_NAME,
    )


async def rerank_candidates(
    candidates: list[dict[str, Any]],
    query: str,
    model: str,          # только для логов: модель зашита в _MODEL_NAME
    model_manager: Any,  # kept for API compatibility, unused
    top_k: int,
) -> list[dict[str, Any]]:
    if not candidates:
        return candidates
    _warn_once_about_ignored_model(model)

    try:
        import asyncio

        loop = asyncio.get_event_loop()
        texts = [item.get("text", "") for item in candidates]
        queries = [query] * len(texts)

        # Run GPU inference in thread pool to avoid blocking event loop
        scores = await loop.run_in_executor(None, _score_batch, queries, texts)

        for item, score in zip(candidates, scores):
            item["reranker_score"] = score

        reranked = sorted(candidates, key=lambda x: x.get("reranker_score", 0.0), reverse=True)
        logger.debug(
            "Qwen3-Reranker: scored %d candidates | top=%.3f bottom=%.3f",
            len(reranked),
            reranked[0]["reranker_score"] if reranked else 0,
            reranked[-1]["reranker_score"] if reranked else 0,
        )
        return reranked[:top_k]

    except Exception as exc:
        logger.warning("Qwen3-Reranker failed, keeping original order: %s", exc)
        return candidates[:top_k]
