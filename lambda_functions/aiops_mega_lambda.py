"""
Key behavior:
- Accepts multipart/form-data with multiple "image" fields + user_id, location, timestamp, damage_description
- Filters images via Rekognition (car labels) + sharpness (Rekognition or local fallback)
- Metadata validation (EXIF vs provided)
- Claude (Bedrock) multi-image damage-description verification
- Titan embeddings per-image -> averaged using NumPy
- OpenSearch k-NN duplicate detection (user-scoped)
- ONLY on VERIFIED claims: save images to S3 under claims/{safe_user}/{claim_id}/, index averaged vector (claim_id) in OpenSearch, persist record in DynamoDB
- On ANY rejection or error before verification: return JSON with is_claim_valid=False and "reason"; do NOT persist
"""

import os
import json
import logging
import uuid
import requests
import boto3
from botocore.exceptions import ClientError

# from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth


# ----------------------------
# Logging
# ----------------------------
LOG = logging.getLogger()
LOG.setLevel(logging.INFO)

# ----------------------------
# Environment / config
# ----------------------------
REGION = os.getenv("REGION", "us-east-1")
# OPENSEARCH_ENDPOINT = os.getenv("OPENSEARCH_ENDPOINT")
# OPENSEARCH_INDEX = os.getenv("OPENSEARCH_INDEX", "claims-index")
# OPENSEARCH_VECTOR_FIELD = os.getenv("OPENSEARCH_VECTOR_FIELD", "vector")

DATADOG_API_KEY = os.getenv("DATADOG_API_KEY")
DATADOG_APP_KEY = os.getenv("DATADOG_APP_KEY")
DATADOG_SITE = os.getenv("DATADOG_SITE", "us5")
SERVICENOW_INSTANCE = os.getenv("SERVICENOW_INSTANCE")
SERVICENOW_USER = os.getenv("SERVICENOW_USER", "")
SERVICENOW_PASS = os.getenv("SERVICENOW_PASS", "")

DATADOG_QUERY_LOG_LIMIT = int(os.getenv("DATADOG_QUERY_LOG_LIMIT", "5"))
BEDROCK_MODEL = os.getenv("BEDROCK_MODEL", "")
MAX_INPUT_CHARS = int(os.getenv("MAX_INPUT_CHARS", "2000"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "5000"))
TITAN_MODEL = os.getenv("TITAN_MODEL", "amazon.titan-embed-image-v1")

# Similarity Threshold and Upload Bucket
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.8"))
# S3_UPLOAD_BUCKET = os.getenv("S3_UPLOAD_BUCKET", "").strip()

# ----------------------------
# AWS clients & OpenSearch
# ----------------------------

session = boto3.Session(region_name=REGION)
bedrock = session.client("bedrock-runtime", region_name=REGION)
# s3_client = session.client("s3")

# OpenSearch auth using AWS sigv4
credentials = session.get_credentials().get_frozen_credentials()
aws_auth = AWS4Auth(
    credentials.access_key,
    credentials.secret_key,
    REGION,
    "aoss",  # Amazon OpenSearch Serverless service name
    session_token=credentials.token,
)

# host = (
#     OPENSEARCH_ENDPOINT.replace("https://", "").replace("http://", "").split("/", 1)[0]
# )  # type: ignore
# _opensearch_client = OpenSearch(
#     hosts=[{"host": host, "port": 443}],
#     http_auth=aws_auth,
#     use_ssl=True,
#     verify_certs=True,
#     connection_class=RequestsHttpConnection,
#     timeout=30,
# )

# -----------------
# HELPER FUNCTIONS
# -----------------


def datadog_query_logs(
    query: str, time_from: str = "now-1d", time_to: str = "now", limit: int = 50
):
    url = f"https://api.{DATADOG_SITE}.datadoghq.com/api/v2/logs/events/search"
    headers = {
        "DD-API-KEY": DATADOG_API_KEY,
        "DD-APPLICATION-KEY": DATADOG_APP_KEY,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    body = {
        "filter": {"from": time_from, "to": time_to, "query": query},
        "page": {"limit": limit},
    }
    resp = requests.post(url, headers=headers, json=body, timeout=10)
    resp.raise_for_status()
    payload = resp.json()
    return payload.get("data", [])


def get_header_value(event, header_name):
    """Case-insensitive header retrieval."""
    headers = event.get("headers") or {}
    for k, v in headers.items():
        if k.lower() == header_name.lower():
            return v
    return None


def servicenow_create_incident(
    short_description: str, description: str, extra: dict = {}
):
    url = f"https://{SERVICENOW_INSTANCE}/api/now/table/incident"
    auth = (SERVICENOW_USER, SERVICENOW_PASS)
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    payload = {"short_description": short_description, "description": description}
    if extra:
        payload.update(extra)
    resp = requests.post(url, auth=auth, headers=headers, json=payload, timeout=10)
    resp.raise_for_status()
    return resp.json().get("result")


# TODO: Add DD_LOGS to Prompt
def call_bedrock_issue_handler(user_input: str, dd_logs: str):
    """
    Send `user_input` to Bedrock model, expect JSON output following schema:
    {
      "issue_valid": true/false,
      "confidence": float (0.0-100.0),
      "analysis": string,
      "suggested_action": string
    }
    """
    safe_input = (
        (user_input or "")[:MAX_INPUT_CHARS].replace('"', '\\"').replace("\n", " ")
    )
    prompt = f"""
You are an issue-handling assistant.

Task:
- You receive as input a string called "user_input" — this is a user-submitted issue or request.
- Your job is to decide:
    1. Is the input valid / meaningful / actionable?
    2. If yes, propose a resolution or next-step action.

Return a JSON object with exactly these keys:
{{
  "issue_valid": true or false,
  "confidence": <number between 0.0 and 100.0>,
  "analysis": "<your human-readable reasoning/explanation>",
  "suggested_action": "<a suggestion or resolution if issue_valid = true, otherwise empty string>"
}}

Constraints:
- Use double quotes for all keys and string values.
- Use lowercase JSON booleans (true, false).
- Do not output any extra text outside the JSON object.

Input:
user_input: "{safe_input}"
"""
    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": MAX_TOKENS,
        "messages": [{"role": "user", "content": prompt}],
    }

    try:
        resp = bedrock.invoke_model(
            modelId=BEDROCK_MODEL,
            body=json.dumps(payload),
            contentType="application/json",
            accept="application/json",
        )
    except ClientError as e:
        LOG.exception("Bedrock invoke_model failed: %s", e)
        raise

    body_stream = resp.get("body")
    if body_stream is None:
        raise RuntimeError("No body in Bedrock response")

    raw = body_stream.read().decode("utf-8").strip()
    LOG.debug("Raw Bedrock response: %s", raw)

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        LOG.error("Failed to parse JSON from Bedrock output: %s", raw)
        raise

    # Validate schema
    for key in ("issue_valid", "confidence", "analysis", "suggested_action"):
        if key not in result:
            raise ValueError(f"Missing key in Bedrock output: {key}")

    return result


# TODO: Review the embedding feature once before using
# def call_titan_embedding(code_str: str):
#     try:
#         body = {"inputText": code_str}
#         resp = bedrock.invoke_model(modelId=TITAN_MODEL, body=json.dumps(body))
#         return json.loads(resp["body"].read()).get("embedding")
#     except Exception:
#         LOG.exception("call_titan_embedding failed")
#         raise

# def opensearch_knn_search(vector, k=5):
#     """
#     Perform a k-NN search on OpenSearch.
#     """
#     if _opensearch_client is None:
#         raise RuntimeError("OpenSearch client not configured")

#     query = {
#         "size": k,
#         "query": {
#             "bool": {
#                 "must": [
#                     {"knn": {OPENSEARCH_VECTOR_FIELD: {"vector": vector, "k": k}}}
#                 ],
#             }
#         },
#     }
#     res = _opensearch_client.search(index=OPENSEARCH_INDEX, body=query)
#     hits = res.get("hits", {}).get("hits", [])
#     return [
#         {"id": h.get("_id"), "score": h.get("_score"), "source": h.get("_source", {})}
#         for h in hits
#     ]


# def opensearch_index_vector(solution_id, vector, extra=None):
#     """Adds a vector to the OPENSEARCH_INDEX"""
#     doc = {"solution_id": solution_id, OPENSEARCH_VECTOR_FIELD: vector}
#     if extra:
#         doc.update(extra)
#     try:
#         return _opensearch_client.index(index=OPENSEARCH_INDEX, body=doc)
#     except Exception as e:
#         LOG.exception("OpenSearch index failed: %s", e)
#         raise


# def ensure_opensearch_index_mapping():
#     """Ensure that the opensearch index and collection exists. If not, create one with the values set in env variables or the default ones."""
#     try:
#         if not _opensearch_client.indices.exists(index=OPENSEARCH_INDEX):
#             LOG.info(f"Creating OpenSearch index '{OPENSEARCH_INDEX}'")
#             _opensearch_client.indices.create(
#                 index=OPENSEARCH_INDEX,
#                 body={
#                     "mappings": {
#                         "properties": {
#                             "claim_id": {"type": "keyword"},
#                             "user_id": {"type": "keyword"},
#                             OPENSEARCH_VECTOR_FIELD: {
#                                 "type": "knn_vector",
#                                 "dimension": 1024,
#                                 "method": {"name": "hnsw", "engine": "faiss"},
#                             },
#                         }
#                     },
#                     "settings": {"index.knn": True},
#                 },
#             )
#     except Exception as e:
#         LOG.error("Failed to ensure OpenSearch index mapping: %s", e)
#         raise


# ---------------------------------
def lambda_handler(event, context):
    """
    Expected event (from API Gateway or other trigger):
    {
      "user_input": "<text>",
      plus possibly HTTP headers accessible via event['headers']
    }
    """
    # solution_id = f"solution_{str(uuid.uuid4())}"
    user_input = (event.get("user_input") or "").strip()

    if not user_input:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": "Bad request: missing 'user_input'"}),
        }

    try:
        # ensure_opensearch_index_mapping()

        # 1. Always query Datadog logs (for analysis / context)
        # Use user_input text as search query; you may refine query logic as needed
        dd_logs = datadog_query_logs(
            query=user_input,
            time_from="now-1d",
            time_to="now",
            limit=DATADOG_QUERY_LOG_LIMIT,
        )
        LOG.info("Datadog logs returned: %d events", len(dd_logs))

        # 2. Call Bedrock / LLM to analyze user_input
        llm_out = call_bedrock_issue_handler(user_input, dd_logs)

        # 3. Build response payload
        response = {
            "issue_valid": llm_out.get("issue_valid"),
            "confidence": llm_out.get("confidence"),
            "analysis": llm_out.get("analysis"),
            "suggested_action": llm_out.get("suggested_action"),
            "datadog_logs": dd_logs,
        }

        # Step: Embeddings
        # neighbors = opensearch_knn_search(vector, user_id, k=5)
        # high_sim_neighbors = [
        #     n for n in neighbors if n.get("score", 0.0) >= SIMILARITY_THRESHOLD
        # ]

        # report["OpenSearch_agent"] = (
        #     (
        #         f"Similar claims found ({len(high_sim_neighbors)}), the Claim IDs and their respective match percentage are as follows: "
        #         + ", ".join(
        #             f"{neighbor['source'].get('claim_id', neighbor['id'])} ({neighbor['score'] * 100:.2f}%)"
        #             for neighbor in high_sim_neighbors
        #         )
        #     )
        #     if high_sim_neighbors
        #     else "No similar claims found."
        # )
        # if high_sim_neighbors:
        #     report["is_claim_valid"] = False
        #     return {"statusCode": 200, "body": json.dumps(report)}

        # opensearch_index_vector(solution_id, vector)

        # return {"statusCode": 200, "body": json.dumps(report)}

        # 4. If issue is valid, optionally create ticket in ServiceNow
        # TODO: Reformat the ServiceNow Ticket template
        if llm_out.get("issue_valid"):
            ticket = servicenow_create_incident(
                short_description=f"Issue: {user_input[:100]}",
                description=f"User input: {user_input}\n\nRelevant logs:\n{json.dumps(dd_logs)[:2000]}",
                extra={},
            )
            response["servicenow_ticket"] = ticket

        return {"statusCode": 200, "body": json.dumps(response)}
    except Exception as e:
        LOG.exception("Internal error")
        return {
            "statusCode": 500,
            "body": json.dumps({"error": f"Internal server error: {e}"}),
        }
