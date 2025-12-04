"""
Key behavior:
- Accepts user_input via API Gateway.
- Queries Datadog for recent 'error' or 'warn' logs.
- Uses Bedrock (LLM) to determine if the user query mentions a known service.
- If no service is mentioned, it returns a generic response (Early Exit).
- If a service is mentioned, it formats the relevant logs and sends them
to the main Bedrock model for analysis and suggested action.
- If the LLM output is valid, it optionally creates a ServiceNow ticket.
"""

import os
import json
import re
import logging
import uuid
import requests
import boto3
from botocore.exceptions import ClientError
# from requests_aws4auth import AWS4Auth
# from opensearchpy import OpenSearch, RequestsHttpConnection


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

DATADOG_QUERY_LOG_LIMIT = int(os.getenv("DATADOG_QUERY_LOG_LIMIT", "50"))
CLAUDE_MODEL = os.getenv(
    "CLAUDE_MODEL", "global.anthropic.claude-haiku-4-5-20251001-v1:0"
)
MAX_INPUT_CHARS = int(os.getenv("MAX_INPUT_CHARS", "2000"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "5000"))
SERVICE_CHECK_MAX_TOKENS = 50
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
# credentials = session.get_credentials().get_frozen_credentials()
# aws_auth = AWS4Auth(
#     credentials.access_key,
#     credentials.secret_key,
#     REGION,
#     "aoss",  # Amazon OpenSearch Serverless service name
#     session_token=credentials.token,
# )

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


def get_header_value(event, header_name):
    """Case-insensitive header retrieval."""
    headers = event.get("headers") or {}
    for k, v in headers.items():
        if k.lower() == header_name.lower():
            return v
    return None


def invoke_claude_model(prompt: str, max_tokens: int) -> dict:
    """
    Generic function to invoke the Claude model via Bedrock for structured JSON output.
    It handles the Boto3 invocation and robust regex-based JSON extraction.
    """
    LOG.info(f"Invoking Claude model: {CLAUDE_MODEL} with max_tokens: {max_tokens}")

    # Content structure for text-only input
    content = [{"type": "text", "text": prompt}]

    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": content}],
    }

    try:
        # 1. Invoke the model
        response = bedrock.invoke_model(
            modelId=CLAUDE_MODEL,
            body=json.dumps(payload),
            contentType="application/json",
            accept="application/json",
        )

        # 2. Read the Boto3 response body and parse the outer JSON envelope
        response_body = json.loads(response["body"].read())

        # 3. Extract the raw text output from the model's response structure
        raw_text_output = response_body["content"][0]["text"].strip()
        LOG.info("Successfully received response from Claude.")

        # 4. Use regex to find and extract the JSON object (robust parsing)
        json_match = re.search(r"\{.*\}", raw_text_output, re.DOTALL)

        if json_match:
            return json.loads(json_match.group(0))
        else:
            LOG.error(
                "Could not find a valid JSON object in the model's response. Raw response: %s",
                raw_text_output,
            )
            raise ValueError("Failed to parse JSON from Claude's response.")

    except Exception as e:
        LOG.exception("An error occurred during the Claude model invocation: %s", e)
        # Re-raise the exception to be handled by the specific caller function
        raise


def datadog_query_logs(
    time_from: str = "now-1d",
    time_to: str = "now",
    limit: int = DATADOG_QUERY_LOG_LIMIT,
):
    """
    Queries Datadog logs for all 'error' or 'warn' statuses within a time frame.
    Query is hardcoded to status:(error OR warn).
    """
    # Hardcoding the query - status: (error OR warn)
    query = "status:(error OR warn)"

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

    try:
        resp = requests.post(url, headers=headers, json=body, timeout=10)
        resp.raise_for_status()
        payload = resp.json()
        return payload.get("data", [])
    except requests.exceptions.RequestException as e:
        LOG.error("An error occurred during Datadog API call: %s", e)
        return []


def extract_unique_services(logs: list[dict]) -> set[str]:
    """
    Parses Datadog log entries (from the 'data' field) to extract all
    unique service names.
    """
    unique_services = set()
    # TODO: When increasing the number of logs to be processed, add multithreading for faster processing.
    for log in logs:
        # Service is typically found in log["attributes"]["service"]
        try:
            service_name = log.get("attributes", {}).get("service")
            if service_name:
                unique_services.add(service_name)
        except Exception:
            # Handle malformed log entries gracefully
            continue
    return unique_services


def format_logs_for_llm(logs: list[dict], selected_service: str) -> str:
    """
    Filters logs for a specific service and formats key fields into a compact string
    suitable for LLM prompting.
    """
    # Filter logs to only include the service the LLM selected
    filtered_logs = [
        log
        for log in logs
        if log.get("attributes", {}).get("service") == selected_service
    ]

    formatted_logs = []
    for log in filtered_logs:
        attrs = log.get("attributes", {})
        # Extract only the most relevant fields to conserve token count
        timestamp = attrs.get("timestamp", "N/A")
        message = (
            attrs.get("message", "No message").split("\n")[0].strip()
        )  # Take only the first line
        status = attrs.get("status", "N/A")

        formatted_logs.append(
            f"[TS: {timestamp}] [Status: {status}] Message: {message}"
        )

    return "\n".join(formatted_logs)


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


def call_bedrock_service_check(user_input: str, unique_services: set) -> str:
    """
    Builds the prompt for service selection and calls the generic LLM function.
    Returns the selected service name or "N/A".
    """
    safe_input = (
        (user_input or "")[:MAX_INPUT_CHARS].replace('"', '\\"').replace("\n", " ")
    )

    service_list_with_fallback = list(unique_services) + ["N/A"]

    prompt = f"""
    You are an AI assistant for routing user issues. Your task is to select 
    the best match between the 'User Request' and the 'Service List'.

    Service List: {service_list_with_fallback}

    User Request: "{safe_input}"

    Instructions:
    Return a JSON object with a single key 'service_key'. 
    The value must be the **exact service name** from the list that is most relevant, 
    or the string **"N/A"** if none are relevant. Do NOT include any markdown or extra text.

    Example Output: {{"service_key": "TR1-app"}}

    Output:
    """

    try:
        result_dict = invoke_claude_model(
            prompt=prompt, max_tokens=SERVICE_CHECK_MAX_TOKENS
        )

        selected_service = result_dict.get("service_key", "N/A").strip()

        # Final validation
        if selected_service in unique_services or selected_service == "N/A":
            return selected_service

        LOG.warning(
            "LLM returned unexpected service key '%s'. Defaulting to N/A.",
            selected_service,
        )
        return "N/A"

    except Exception as e:
        LOG.error("Service check failed during invocation or parsing: %s", e)
        return "N/A"


def call_bedrock_issue_handler(user_input: str, dd_log_context: str) -> dict:
    """
    Builds the prompt for full issue analysis and calls the generic LLM function.
    Returns the structured analysis dictionary.
    """
    safe_input = (
        (user_input or "")[:MAX_INPUT_CHARS].replace('"', '\\"').replace("\n", " ")
    )

    prompt = f"""
You are an issue-handling assistant.

Task:
- You receive a user-submitted issue ("user_input") and a summary of relevant 
  recent error/warn logs ("datadog_context") for the implied service.
- Decide if the input is valid/actionable and propose a resolution based on the logs.

Return a JSON object with exactly these keys:
{{
  "issue_valid": true or false,
  "confidence": <number between 0.0 and 100.0>,
  "analysis": "<your human-readable reasoning/explanation>",
  "suggested_action": "<a suggestion or resolution if issue_valid = true, otherwise empty string>"
}}

Constraints:
- Your SOLE output must be a single, valid JSON object.
- Use double quotes for all keys and string values.
- Use lowercase JSON booleans (true, false).
- Do not output any extra text, explanations, or markdown outside the JSON object.

Input:
user_input: "{safe_input}"
datadog_context: 
---
{dd_log_context}
---
"""
    try:
        # Use the global MAX_TOKENS for the main analysis
        result = invoke_claude_model(prompt=prompt, max_tokens=MAX_TOKENS)

    except Exception as e:
        LOG.error("Main issue handler failed during invocation or parsing: %s", e)
        # Return graceful error structure on failure
        result = {
            "issue_valid": False,
            "confidence": 0.0,
            "analysis": f"Bedrock invocation or parsing failed: {e}",
            "suggested_action": "",
        }

    # Final structure validation: ensure all expected keys are present
    final_result = {}
    for key in ("issue_valid", "confidence", "analysis", "suggested_action"):
        final_result[key] = result.get(key)

    return final_result


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
    Main entry point for the Lambda function.
    """
    user_input = ""
    try:
        if "body" in event and event["body"]:
            # Decode the JSON string in the 'body' field and extract user input
            body_data = json.loads(event["body"])
            user_input = (body_data.get("user_input") or "").strip()

            if not user_input:
                return {
                    "statusCode": 400,
                    "body": json.dumps(
                        {
                            "error": f"Bad request: missing 'user_input', got the following input: {event['body']}"
                        }
                    ),
                }

        # ensure_opensearch_index_mapping()

        # 1. Always query Datadog logs for a list of recent error/warn events
        dd_logs_raw = datadog_query_logs()
        LOG.info("Datadog logs returned: %d events", len(dd_logs_raw))

        # 2. Extract unique service names from all logs
        unique_services = extract_unique_services(dd_logs_raw)
        LOG.info("Unique services found: %s", list(unique_services))

        # 3. Call Bedrock to check if the user input mentions any known service
        selected_service = call_bedrock_service_check(user_input, unique_services)
        LOG.info("Bedrock Service Check Result: %s", selected_service)

        # 4. Branching Logic (Early Exit if service is 'N/A')
        if selected_service == "N/A":
            # C1. Handle case where NO services were found OR user was generic
            error_message = "Your request is too general."
            if unique_services:
                error_message += f" Please specify one of the following services mentioned in recent error/warn logs: {list(unique_services)}"
            else:
                error_message += (
                    " No recent error/warn logs were found to provide context."
                )

            return {
                "statusCode": 200,
                "body": json.dumps(
                    {
                        "issue_valid": False,
                        "analysis": error_message,
                        "suggested_action": "Request user to specify the application or service name.",
                        "datadog_logs_count": len(dd_logs_raw),
                        "servicenow_ticket": None,
                    }
                ),
            }
        # 5. Format the logs for the main LLM call (filtered by the selected service)
        dd_log_context = format_logs_for_llm(dd_logs_raw, selected_service)

        # 6. Call the main Bedrock LLM to analyze the issue with log context
        llm_out = call_bedrock_issue_handler(user_input, dd_log_context)

        # 7. Build response payload
        response = {
            "issue_valid": llm_out.get("issue_valid"),
            "confidence": llm_out.get("confidence"),
            "analysis": llm_out.get("analysis"),
            "suggested_action": llm_out.get("suggested_action"),
            "target_service": selected_service,
            "datadog_logs_context": dd_log_context,
        }
        LOG.info("Response payload: %s", response)

        # 8. If issue is valid, create ticket in ServiceNow
        if llm_out.get("issue_valid"):
            # Use LLM analysis and suggested action for the ticket description
            ticket_desc = f"User input: {user_input}\n\nAI Agent's Analysis:\n{llm_out.get('analysis')}\n\nAI Agent's Suggested Action:\n{llm_out.get('suggested_action')}\n\nRelevant Logs:\n{dd_log_context}"
            ticket = servicenow_create_incident(
                short_description=f"[{selected_service}] Issue: {user_input}",
                description=ticket_desc,
                extra={"category": "LLM-Assisted Resolution"},
            )
            response["servicenow_ticket"] = ticket
        else:
            response["servicenow_ticket"] = None

        return {"statusCode": 200, "body": json.dumps(response)}
    except Exception as e:
        LOG.exception("Internal error during Lambda execution")
        return {
            "statusCode": 500,
            "body": json.dumps({"error": f"Internal server error: {e}"}),
        }
