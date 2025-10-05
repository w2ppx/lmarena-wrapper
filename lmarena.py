import uuid
import json
import ast
import sys
import random
from pathlib import Path
from curl_cffi import requests

DEFAULT_COOKIES = {
    'cf_clearance': '',
    'arena-auth-prod-v1.0': '',
    'arena-auth-prod-v1.1': '',
    'cookie-preferences': '',
    'arena-auth-prod-v1-code-verifier': '',
    '__cf_bm': '',
}

USER_AGENT_IMPERSONATE_MAP = {
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36": "chrome131",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36": "chrome136",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36": "chrome133a",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Safari/605.1.15": "safari180",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.4 Safari/605.1.15": "safari184",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:135.0) Gecko/20100101 Firefox/135.0": "firefox135",
}

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.5",
    "Referer": "https://lmarena.ai/?mode=direct",
    "Content-Type": "text/plain;charset=UTF-8",
    "Origin": "https://lmarena.ai",
    "Sec-GPC": "1",
    "Connection": "keep-alive",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-origin",
    "Priority": "u=0",
}


def _extract_models_recursive(node):
    """Recursively extract models from nested data structure"""
    if isinstance(node, dict):
        if "initialModels" in node and isinstance(node["initialModels"], list):
            return node["initialModels"]
        for value in node.values():
            result = _extract_models_recursive(value)
            if result:
                return result
    elif isinstance(node, list):
        for item in node:
            result = _extract_models_recursive(item)
            if result:
                return result
    return None


def get_models(models_file="models.json"):
    """
    Parse and extract models from models.json file.

    Args:
        models_file: Path to the models.json file (default: 'models.json')

    Returns:
        List of model dictionaries with 'id', 'label', and 'label_key' fields

    Raises:
        FileNotFoundError: If models file doesn't exist
        ValueError: If file format is invalid or no models found
    """
    raw_content = Path(models_file).read_text(encoding="utf-8").strip()
    prefix = "self.__next_f.push("
    if not raw_content.startswith(prefix) or not raw_content.endswith(")"):
        raise ValueError("Unexpected models.json format")

    inner_literal = raw_content[len(prefix) : -1]

    try:
        payload_wrapper = ast.literal_eval(inner_literal)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"Failed to evaluate models payload: {exc}")

    if not isinstance(payload_wrapper, (list, tuple)):
        raise ValueError("Unexpected payload wrapper type")

    stream_entries = None
    for item in payload_wrapper:
        if isinstance(item, str) and item.startswith("5:"):
            stream_entries = item[2:]
            break

    if stream_entries is None:
        raise ValueError("Stream data not found in models payload")

    try:
        stream_entries = json.loads(stream_entries)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse stream entries: {exc}")

    if not isinstance(stream_entries, list):
        raise ValueError("Unexpected payload structure")

    models_raw = []
    for entry in stream_entries:
        if isinstance(entry, (list, dict)):
            result = _extract_models_recursive(entry)
            if result:
                models_raw = result
                break

    if not models_raw:
        raise ValueError("No models found in payload")

    label_keys = ("publicName", "displayName", "name", "title", "label")
    models = []
    for entry in models_raw:
        model_id = entry.get("id")
        if not model_id:
            continue

        label_key = None
        label_value = None
        for candidate in label_keys:
            value = entry.get(candidate)
            if isinstance(value, str) and value:
                label_key = candidate
                label_value = value
                break

        if label_key is None:
            continue

        capabilities = entry.get("capabilities", {}) or {}
        output_capabilities = capabilities.get("outputCapabilities", {}) or {}
        supports_text = bool(output_capabilities.get("text") is True)
        supports_image = bool(output_capabilities.get("image"))

        models.append(
            {
                "id": model_id,
                "label_key": label_key,
                "label": label_value,
                "supports_text": supports_text,
                "supports_image": supports_image,
            }
        )

    if not models:
        raise ValueError("No usable models extracted")

    return models


def _get_impersonate_for_user_agent(user_agent):
    """
    Get the appropriate impersonate value for a given User-Agent string.

    Args:
        user_agent: The User-Agent string from headers

    Returns:
        The impersonate value to use, or 'chrome131' as fallback
    """
    return USER_AGENT_IMPERSONATE_MAP.get(user_agent, "chrome131")


def _parse_streaming_response(response_text):
    """
    Parse streaming response format with thinking (ag:), text (a0:), and metadata (ad:).

    Returns:
        Tuple of (thinking, response_text, metadata)
    """
    lines = response_text.strip().split("\n")
    thinking = []
    response = []
    metadata = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith("ag:"):
            try:
                text_part = line[3:]
                text = json.loads(text_part)
                thinking.append(text)
            except (json.JSONDecodeError, ValueError):
                continue

        elif line.startswith("a0:"):
            try:
                text_part = line[3:]
                text = json.loads(text_part)
                response.append(text)
            except (json.JSONDecodeError, ValueError):
                continue

        elif line.startswith("ad:"):
            try:
                metadata_part = line[3:]
                metadata = json.loads(metadata_part)
            except (json.JSONDecodeError, ValueError):
                continue

    return "".join(thinking), "".join(response), metadata


def _parse_image_streaming_response(response_text):
    """
    Parse image streaming response format with images (a2:[...]) and metadata (ad:{...}).

    Returns:
        Tuple of (images, metadata) where images is a list of dicts with keys like
        'type', 'image' (URL), and 'mimeType'.
    """
    lines = response_text.strip().split("\n")
    images = []
    metadata = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith("a2:"):
            try:
                arr_part = line[3:]
                arr = json.loads(arr_part)
                if isinstance(arr, list):
                    for item in arr:
                        if isinstance(item, dict):
                            images.append(item)
            except (json.JSONDecodeError, ValueError):
                continue

        elif line.startswith("ad:"):
            try:
                metadata_part = line[3:]
                metadata = json.loads(metadata_part)
            except (json.JSONDecodeError, ValueError):
                continue

    return images, metadata


def refresh_token(cookies=None, headers=None):
    """Refresh authentication cookies by going to lmarena.ai with outdated cookies.

    Args:
        cookies: Optional existing cookies to send (defaults to DEFAULT_COOKIES)
        headers: Optional headers to send (defaults to DEFAULT_HEADERS)

    Returns:
        dict: Merged/updated cookies to use on subsequent requests

    """
    base_cookies = cookies or DEFAULT_COOKIES
    req_headers = headers.copy() if headers is not None else DEFAULT_HEADERS.copy()
    impersonate = _get_impersonate_for_user_agent(req_headers["User-Agent"])

    resp = requests.get(
        "https://lmarena.ai",
        cookies=base_cookies,
        headers=req_headers,
        impersonate=str(impersonate),
        timeout=30,
    )

    merged = dict(base_cookies)
    if getattr(resp, "status_code", None) == 200:
        try:
            if hasattr(resp, "cookies") and resp.cookies:
                if hasattr(resp.cookies, "get_dict"):
                    merged.update(resp.cookies.get_dict())
                elif isinstance(resp.cookies, dict):
                    merged.update(resp.cookies)
                else:
                    try:
                        new_cookies = {c.name: c.value for c in resp.cookies}
                        merged.update(new_cookies)
                    except Exception:
                        pass
        except Exception:
            pass
    return merged


def ask(model_id, message, cookies=None, headers=None):
    """
    Send a message to a specific model and get the response with fallback impersonation.

    Args:
        model_id: The ID of the model to use
        message: The message to send
        cookies: Optional custom cookies (uses DEFAULT_COOKIES if not provided)
        headers: Optional custom headers (uses DEFAULT_HEADERS if not provided)

    Returns:
        Dictionary with 'thinking', 'response', and 'metadata' keys

    Example:
        result = ask('e2d9d353-6dbe-4414-bf87-bd289d523726', 'Hello!')
        print(result['response'])  # The model's response
        print(result['thinking'])  # The model's reasoning (if available)
    """
    import logging

    logger = logging.getLogger(__name__)

    logger.info(f"ask() called with model_id: {model_id}, message: {message[:100]}...")
    using_default_cookies = cookies is None
    if cookies is None:
        cookies = DEFAULT_COOKIES
    if headers is None:
        headers = DEFAULT_HEADERS.copy()

    def build_payload():
        sid = str(uuid.uuid4())
        uid = str(uuid.uuid4())
        aid = str(uuid.uuid4())
        payload = (
            f'{{"id":"{sid}","mode":"direct","modelAId":"{model_id}",'
            f'"userMessageId":"{uid}","modelAMessageId":"{aid}","messages":[{{"id":"{uid}","role":"user","content":"{message}",'
            f'"experimental_attachments":[],"parentMessageIds":[],"participantPosition":"a","modelId":null,"evaluationSessionId":"{sid}","status":"pending","failureReason":null}},'
            f'{{"id":"{aid}","role":"assistant","content":"","reasoning":"","experimental_attachments":[],"parentMessageIds":["{uid}"],"participantPosition":"a","modelId":"{model_id}","evaluationSessionId":"{sid}","status":"pending","failureReason":null}}],"modality":"chat"}}'
        )
        return sid, uid, aid, payload

    user_agent = headers.get("User-Agent", DEFAULT_HEADERS["User-Agent"])
    impersonate = _get_impersonate_for_user_agent(user_agent)

    logger.info(f"Using impersonate: {impersonate}")

    available_impersonates = list(USER_AGENT_IMPERSONATE_MAP.values())
    random.shuffle(available_impersonates)

    impersonates_to_try = [impersonate] + [
        imp for imp in available_impersonates if imp != impersonate
    ]

    logger.info(f"Will try impersonates in order: {impersonates_to_try}")

    last_exception = None

    for i, current_impersonate in enumerate(impersonates_to_try):
        try:
            logger.info(
                f"Trying impersonate {i + 1}/{len(impersonates_to_try)}: {current_impersonate}"
            )

            max_session_retries = 3
            last_resp = None
            for attempt in range(max_session_retries):
                session_id, user_msg_id, model_a_msg_id, data = build_payload()
                logger.info(
                    f"Making HTTP request to lmarena.ai... (session attempt {attempt + 1}/{max_session_retries})"
                )
                response = requests.post(
                    "https://lmarena.ai/nextjs-api/stream/create-evaluation",
                    cookies=cookies,
                    headers=headers,
                    data=data,
                    impersonate=current_impersonate,
                    timeout=30,
                )
                last_resp = response
                logger.info(f"Response status code: {response.status_code}")

                try:
                    err_payload = json.loads(response.text)
                    if isinstance(err_payload, dict) and isinstance(
                        err_payload.get("error"), str
                    ):
                        err_msg = err_payload.get("error")
                        if err_msg == "Cannot select private models in non-battle mode":
                            logger.warning(
                                "Received non-battle private model error; aborting impersonate fallbacks"
                            )
                            raise Exception(
                                "Cannot select private models in non-battle mode"
                            )
                        if (
                            "Evaluation session" in err_msg
                            and "already exists" in err_msg
                        ):
                            logger.warning(
                                f"{err_msg} — regenerating UUIDs and retrying"
                            )
                            continue
                except json.JSONDecodeError:
                    pass

                if response.status_code == 401:
                    logger.warning("401 Unauthorized; refreshing cookies and retrying")
                    try:
                        cookies = refresh_token(cookies=cookies, headers=headers)
                        if using_default_cookies and isinstance(DEFAULT_COOKIES, dict):
                            try:
                                DEFAULT_COOKIES.update(cookies)
                            except Exception:
                                pass
                    except Exception as refresh_exc:
                        logger.warning(f"Token refresh failed: {refresh_exc}")
                    continue

                break

            try:
                err_payload = json.loads(response.text)
                if (
                    isinstance(err_payload, dict)
                    and err_payload.get("error")
                    == "Cannot select private models in non-battle mode"
                ):
                    logger.warning(
                        "Received non-battle private model error; aborting impersonate fallbacks"
                    )
                    raise Exception("Cannot select private models in non-battle mode")
            except json.JSONDecodeError:
                pass

            if response.status_code == 200:
                logger.info(f"Request successful with {current_impersonate}")
                thinking, response_text, metadata = _parse_streaming_response(
                    response.text
                )

                logger.info(
                    f"Response parsed - thinking: {len(thinking)} chars, response: {len(response_text)} chars"
                )

                return {
                    "thinking": thinking,
                    "response": response_text,
                    "metadata": metadata,
                }
            else:
                logger.warning(
                    f"Request failed with status {response.status_code}, trying next impersonate"
                )
                continue

        except Exception as e:
            if "Cannot select private models in non-battle mode" in str(e):
                logger.warning(
                    "Non-battle private model error encountered; stopping fallbacks"
                )
                raise
            logger.warning(f"Exception with {current_impersonate}: {e}")
            print(e)
            last_exception = e
            continue

    logger.error(f"All impersonates failed. Last exception: {last_exception}")
    if last_exception:
        raise last_exception
    else:
        raise Exception("All impersonate attempts failed")


def list_models(models, include_ids=False, model_type=None):
    """
    Return a list of formatted model labels, optionally including IDs.

    Args:
        models: List of model dictionaries from get_models()
        include_ids: Include model IDs in the output
        model_type: Optional filter: 'text' or 'image'
    Returns:
        List of strings for display
    """
    if model_type is not None:
        mt = str(model_type).strip().lower()
        if mt not in ("text", "image"):
            raise ValueError("model_type must be 'text' or 'image' if provided")
        if mt == "text":
            models = [m for m in models if m.get("supports_text")]
        else:
            models = [m for m in models if m.get("supports_image")]

    lines = []
    for idx, model in enumerate(models, 1):
        if include_ids:
            lines.append(f"{idx:3d}. {model['label']} ({model['id']})")
        else:
            lines.append(f"{idx:3d}. {model['label']}")
    return lines


def generate_image(model_id, prompt, cookies=None, headers=None):
    """
    Generate an image via LM Arena image modality.

    Args:
        model_id: Image-capable model ID
        prompt: Text prompt to generate an image
        cookies/headers: Optional overrides

    Returns:
        Raw response text (logged) and status code
    """
    import logging

    logger = logging.getLogger(__name__)

    using_default_cookies = cookies is None
    if cookies is None:
        cookies = DEFAULT_COOKIES
    if headers is None:
        headers = DEFAULT_HEADERS.copy()

    def build_payload():
        sid = str(uuid.uuid4())
        uid = str(uuid.uuid4())
        aid = str(uuid.uuid4())
        payload = (
            "{"
            f'"id":"{sid}",'
            '"mode":"direct",'
            f'"modelAId":"{model_id}",'
            f'"userMessageId":"{uid}",'
            f'"modelAMessageId":"{aid}",'
            '"messages":[{'
            f'"id":"{uid}","role":"user","content":"{prompt}","experimental_attachments":[],"parentMessageIds":[],"participantPosition":"a","modelId":null,"evaluationSessionId":"{sid}","status":"pending","failureReason":null'
            "},{"
            f'"id":"{aid}","role":"assistant","content":"","reasoning":"","experimental_attachments":[],"parentMessageIds":["{uid}"],"participantPosition":"a","modelId":"{model_id}","evaluationSessionId":"{sid}","status":"pending","failureReason":null'
            "}],"
            '"modality":"image"'
            "}"
        )
        return sid, uid, aid, payload

    user_agent = headers.get("User-Agent", DEFAULT_HEADERS["User-Agent"])
    primary_imp = _get_impersonate_for_user_agent(user_agent)
    fallback_imps = list(USER_AGENT_IMPERSONATE_MAP.values())
    random.shuffle(fallback_imps)
    order = [primary_imp] + [x for x in fallback_imps if x != primary_imp]

    last_exc = None
    for imp in order:
        try:
            max_session_retries = 3
            for attempt in range(max_session_retries):
                session_id, user_msg_id, model_a_msg_id, data = build_payload()
                resp = requests.post(
                    "https://lmarena.ai/nextjs-api/stream/create-evaluation",
                    cookies=cookies,
                    headers=headers,
                    data=data,
                    impersonate=imp,
                    timeout=100,
                )
                logger.info(
                    f"image gen status={resp.status_code} via {imp} (session attempt {attempt + 1}/{max_session_retries})"
                )

                try:
                    err_payload = json.loads(resp.text)
                    if isinstance(err_payload, dict) and isinstance(
                        err_payload.get("error"), str
                    ):
                        err_msg = err_payload.get("error")
                        if err_msg == "Cannot select private models in non-battle mode":
                            logger.warning("non-battle private model error; aborting")
                            raise Exception(
                                "Cannot select private models in non-battle mode"
                            )
                        if (
                            "Evaluation session" in err_msg
                            and "already exists" in err_msg
                        ):
                            logger.warning(
                                "session already exists, regenerating UUIDs and retrying"
                            )
                            continue
                except json.JSONDecodeError:
                    pass

                if resp.status_code == 401:
                    try:
                        cookies = refresh_token(cookies=cookies, headers=headers)
                        if using_default_cookies and isinstance(DEFAULT_COOKIES, dict):
                            try:
                                DEFAULT_COOKIES.update(cookies)
                            except Exception:
                                pass
                        logger.warning("401 Unauthorized during image gen; refreshed cookies and retrying")
                    except Exception as refresh_exc:
                        logger.warning(f"Token refresh failed during image gen: {refresh_exc}")
                    continue

                images, meta = _parse_image_streaming_response(resp.text)
                logger.info(f"image gen parsed images={len(images)} meta={meta}")
                return {
                    "status_code": resp.status_code,
                    "images": images,
                    "metadata": meta,
                    "raw": resp.text,
                }
        except Exception as e:
            last_exc = e
            continue

    if last_exc:
        raise last_exc
    raise Exception("image generation failed")

