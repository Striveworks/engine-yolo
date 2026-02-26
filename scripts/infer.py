#!/usr/bin/env python3
"""
infer.py — Send multiple images as DISTINCT inputs (one tensor per image).

Each image is sent as its own input:
  image-0, image-1, ..., each with shape [1].

Usage:
  python infer.py cat.png dog.png
  python infer.py --url http://host:port/infer cat.png
  python infer.py --request-id abc123 cat.png dog.png
  TOKEN=abc123 python infer.py --url http://host:port/infer cat.png
"""

from __future__ import annotations

import argparse
import base64
import copy
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_URL = "http://127.0.0.1:8080/infer"
ACTION = "detect"  # set to None to omit parameters


def b64_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send multiple images as distinct KServe V2 inputs"
    )
    parser.add_argument(
        "images",
        metavar="IMAGE",
        nargs="+",
        help="Image file paths",
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help=f"Inference endpoint (default: {DEFAULT_URL})",
    )
    parser.add_argument(
        "--request-id",
        help="If set, add X-Request-ID header with this value",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    paths = [Path(p) for p in args.images]
    for p in paths:
        if not p.is_file():
            print(f"Error: file not found: {p}", file=sys.stderr)
            return 2

    inputs = []
    for i, p in enumerate(paths):
        inp: dict = {
            "name": f"image-{i}",
            "datatype": "BYTES",
            "shape": [1],
            "data": [b64_image(p)],
        }
        if ACTION is not None:
            inp["parameters"] = {"action": ACTION}
        inputs.append(inp)

    payload = {"inputs": inputs}
    body = json.dumps(payload).encode("utf-8")

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    token = os.getenv("TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    if args.request_id:
        headers["X-Request-ID"] = args.request_id

    req = urllib.request.Request(
        args.url,
        data=body,
        method="POST",
        headers=headers,
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            resp_body = resp.read()
            try:
                obj = json.loads(resp_body.decode("utf-8"))
                print_request_response(payload, obj)
            except Exception:
                sys.stdout.buffer.write(resp_body)
                if not resp_body.endswith(b"\n"):
                    sys.stdout.write("\n")
            return 0

    except urllib.error.HTTPError as e:
        err_body = e.read()
        print(f"HTTP {e.code} {e.reason}", file=sys.stderr)
        if err_body:
            try:
                obj = json.loads(err_body.decode("utf-8"))
                print(json.dumps(obj, indent=2), file=sys.stderr)
            except Exception:
                sys.stderr.buffer.write(err_body)
                if not err_body.endswith(b"\n"):
                    sys.stderr.write("\n")
        return 1

    except urllib.error.URLError as e:
        print(f"Request failed: {e}", file=sys.stderr)
        return 1


def print_request_response(req_payload, resp_payload):
    print("Request Payload:")
    req_payload = copy.deepcopy(req_payload)
    for i in range(len(req_payload["inputs"])):
        for j in range(len(req_payload["inputs"][i]["data"])):
            req_payload["inputs"][i]["data"][j] = "<base64-encoded-image>"
    print("```json")
    print(json.dumps(req_payload, indent=2))
    print("```")
    print()
    print("Response Payload:")
    print("```json")
    print(json.dumps(resp_payload, indent=2))
    print("```")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
