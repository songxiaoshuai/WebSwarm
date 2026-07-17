import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from dotenv import load_dotenv
from searchers import SearcherType
from transformers import AutoTokenizer

load_dotenv(override=False)

script_env_path = os.path.join(current_dir, ".env")
load_dotenv(dotenv_path=script_env_path, override=False)

import argparse
import re
from functools import lru_cache
from flask import Flask, request, jsonify


_FRONT_MATTER_RE = re.compile(r"^---\n(.*?)\n---\n?(.*)$", re.DOTALL)
_META_LINE_RE = re.compile(r"^\s*([^:\n]+):\s*(.*?)\s*$")
_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]|[a-zA-Z0-9_]+")
_SENT_SPLIT_RE = re.compile(r"(?<=[。！？!?\.])\s+|\n+")


def _truncate_with_tokenizer(text, tokenizer, snippet_max_tokens):
    if not tokenizer or not snippet_max_tokens or snippet_max_tokens <= 0:
        return text
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= snippet_max_tokens:
        return text
    return tokenizer.decode(tokens[:snippet_max_tokens], skip_special_tokens=True)


def _tokenize_for_match(text):
    return _TOKEN_RE.findall((text or "").lower())


@lru_cache(maxsize=20000)
def _parse_document_text(raw_text):
    if not raw_text:
        return "", "", "", []

    title = ""
    date = ""
    body = raw_text

    match = _FRONT_MATTER_RE.match(raw_text)
    if match:
        meta_block = match.group(1)
        body = match.group(2).strip()
        for line in meta_block.splitlines():
            m = _META_LINE_RE.match(line)
            if not m:
                continue
            key = m.group(1).strip().lower()
            val = m.group(2).strip()
            if key == "title":
                title = val
            elif key == "date":
                date = val

    sentences = []
    for part in _SENT_SPLIT_RE.split(body):
        seg = part.strip()
        if seg:
            sentences.append(seg)
    if not sentences and body.strip():
        sentences = [body.strip()]

    return title, date, body, sentences


def _pick_semantic_window(query, title, sentences):
    q_tokens = _tokenize_for_match(query)
    if not sentences:
        return title or ""

    if not q_tokens:
        return sentences[0]

    q_set = set(q_tokens)
    best_text = sentences[0]
    best_score = -1.0

    # Windowed scoring approximates search-engine snippets: pick a local context
    # where query intent terms are most concentrated, with title alignment bonus.
    for i in range(len(sentences)):
        for window in (1, 2, 3):
            end = i + window
            if end > len(sentences):
                continue
            chunk = " ".join(sentences[i:end]).strip()
            if not chunk:
                continue
            tks = _tokenize_for_match(chunk)
            if not tks:
                continue

            tk_set = set(tks)
            overlap = sum(1 for t in q_set if t in tk_set)
            tf = sum(tks.count(t) for t in q_set)
            exact_bonus = 2.0 if query.lower() in chunk.lower() else 0.0
            title_bonus = 0.0
            if title:
                title_low = title.lower()
                title_hit = sum(1 for t in q_set if t in title_low)
                title_bonus = 0.2 * title_hit

            density = overlap / (len(tks) ** 0.5)
            score = overlap * 3.0 + tf * 0.5 + density + exact_bonus + title_bonus

            if score > best_score:
                best_score = score
                best_text = chunk

    return best_text


def _build_semantic_snippet(query, raw_text, tokenizer, snippet_max_tokens):
    title, date, _, sentences = _parse_document_text(raw_text)
    best_window = _pick_semantic_window(query, title, sentences)

    prefix = ""
    if title:
        prefix = title
    if date:
        prefix = f"{prefix} ({date})" if prefix else date

    snippet = best_window if not prefix else f"{prefix} - {best_window}"
    return _truncate_with_tokenizer(snippet, tokenizer, snippet_max_tokens)


def _do_search(searcher, tokenizer, snippet_max_tokens, default_k, query, k=None):
    num = k if k is not None else default_k
    candidates = searcher.search(query, num)

    for cand in candidates:
        text = cand.get("text", "")
        cand["snippet"] = _build_semantic_snippet(
            query=query,
            raw_text=text,
            tokenizer=tokenizer,
            snippet_max_tokens=snippet_max_tokens,
        )

    results = []
    for cand in candidates:
        text = cand.get("text", "")
        title, date, _, _ = _parse_document_text(text)
        if cand.get("score") is None:
            results.append({"docid": cand["docid"], "snippet": cand["snippet"], "title": title})
        else:
            results.append(
                {
                    "docid": cand["docid"],
                    "score": cand["score"],
                    "snippet": cand["snippet"],
                    "url": cand.get("url", "URL not found"),
                    "title": title,
                    "date": date,
                }
            )
    return results


def main():
    try:
        parser = argparse.ArgumentParser(description="HTTP retrieval server")

        parser.add_argument(
            "--searcher-type",
            choices=SearcherType.get_choices(),
            required=True,
            help=f"Type of searcher to use: {', '.join(SearcherType.get_choices())}",
        )
        parser.add_argument(
            "--snippet-max-tokens",
            type=int,
            default=64,
            help="Maximum snippet length in tokens after semantic extraction (default: 64). Set to -1 to disable truncation.",
        )
        parser.add_argument(
            "--k",
            type=int,
            default=5,
            help="Default number of search results to return (default: 5). Can be overridden per request.",
        )
        parser.add_argument(
            "--get-document",
            action="store_true",
            help="If set, also expose the POST /document endpoint.",
        )
        parser.add_argument(
            "--port",
            type=int,
            default=8080,
            help="Port to bind the HTTP server (default: 8080).",
        )
        parser.add_argument(
            "--server-backend",
            type=str,
            default="waitress",
            choices=["waitress", "flask"],
            help="HTTP backend. waitress is recommended for high concurrency.",
        )
        parser.add_argument(
            "--max-concurrency",
            type=int,
            default=400,
            help="Target max concurrent requests. Used as waitress thread count or flask thread budget.",
        )
        parser.add_argument(
            "--connection-limit",
            type=int,
            default=1200,
            help="Maximum simultaneous open connections for waitress (default: 1200).",
        )
        parser.add_argument(
            "--channel-timeout",
            type=int,
            default=30,
            help="Idle connection timeout in seconds for waitress (default: 30).",
        )
        parser.add_argument(
            "--public",
            action="store_true",
            help="If set, automatically create an ngrok tunnel and print the public URL.",
        )
        parser.add_argument(
            "--hf-token",
            type=str,
            help="Hugging Face token for accessing private datasets/models.",
        )
        parser.add_argument(
            "--hf-home",
            type=str,
            help="Hugging Face home directory for caching models and datasets.",
        )

        temp_args, _ = parser.parse_known_args()

        searcher_class = SearcherType.get_searcher_class(temp_args.searcher_type)
        searcher_class.parse_args(parser)

        args = parser.parse_args()

        if args.hf_token:
            print(f"[DEBUG] Setting HF token from CLI argument: {args.hf_token[:10]}...")
            os.environ["HF_TOKEN"] = args.hf_token
            os.environ["HUGGINGFACE_HUB_TOKEN"] = args.hf_token

        if args.hf_home:
            print(f"[DEBUG] Setting HF home from CLI argument: {args.hf_home}")
            os.environ["HF_HOME"] = args.hf_home

        searcher = searcher_class(args)

        tokenizer = None
        if args.snippet_max_tokens and args.snippet_max_tokens > 0:
            tokenizer = AutoTokenizer.from_pretrained("/mmu_nlp_hdd/wangzhongyuan03/base_models/Qwen3-Embedding-8B")

        app = Flask(__name__)

        @app.route("/search", methods=["POST"])
        def search():
            body = request.get_json(force=True, silent=True) or {}
            q = body.get("q", "").strip()
            if not q:
                return jsonify({"error": "missing or empty field 'q'"}), 400
            k_override = body.get("k", None)
            results = _do_search(
                searcher, tokenizer, args.snippet_max_tokens, args.k, q, k_override
            )
            return jsonify({"results": results})

        if args.get_document:

            @app.route("/document", methods=["POST"])
            def get_document():
                body = request.get_json(force=True, silent=True) or {}
                docid = body.get("url", "").strip()
                if not docid:
                    return jsonify({"error": "missing or empty field 'url'"}), 400
                doc = searcher.get_document(docid)
                if doc is None:
                    return jsonify({"error": f"document not found"}), 404
                return jsonify(doc)

        endpoints = ["POST /search"]
        if args.get_document:
            endpoints.append("POST /document")

        print(
            f"HTTP server started with {searcher.search_type} search "
            f"(snippet_max_tokens={args.snippet_max_tokens}, k={args.k}, "
            f"backend={args.server_backend}, max_concurrency={args.max_concurrency})"
        )
        print(f"Endpoints: {', '.join(endpoints)}")

        if args.public:
            try:
                from pyngrok import ngrok
                from pyngrok.exception import PyngrokNgrokError

                token = os.getenv("NGROK_AUTHTOKEN")
                if token:
                    ngrok.set_auth_token(token)

                try:
                    tunnel = ngrok.connect(addr=args.port, bind_tls=True)
                    doc_line = (
                        f"  Document: POST {tunnel.public_url}/document\n"
                        if args.get_document
                        else ""
                    )
                    print(
                        "\n=============================================\n"
                        f"Public HTTP endpoint: {tunnel.public_url}\n"
                        f"  Search:   POST {tunnel.public_url}/search\n"
                        f"{doc_line}"
                        "=============================================\n"
                    )
                except PyngrokNgrokError as e:
                    first_line = str(e).split("\n")[0]
                    print(
                        f"[Warning] Failed to start ngrok tunnel: {first_line}\n"
                        f"Continuing with local server on http://localhost:{args.port}",
                    )
            except ImportError:
                print("[Warning] pyngrok not installed; continuing without public tunnel.")

        if args.server_backend == "waitress":
            try:
                from waitress import serve

                print(
                    f"Starting waitress on 0.0.0.0:{args.port} "
                    f"with threads={args.max_concurrency}, "
                    f"connection_limit={max(args.connection_limit, args.max_concurrency)}"
                )
                serve(
                    app,
                    host="0.0.0.0",
                    port=args.port,
                    threads=max(8, args.max_concurrency),
                    connection_limit=max(args.connection_limit, args.max_concurrency),
                    channel_timeout=max(5, args.channel_timeout),
                )
            except ImportError:
                print(
                    "[Warning] waitress is not installed. Falling back to Flask threaded server. "
                    "Install waitress for reliable high-concurrency serving: pip install waitress"
                )
                app.run(
                    host="0.0.0.0",
                    port=args.port,
                    threaded=True,
                    debug=False,
                    use_reloader=False,
                )
        else:
            app.run(
                host="0.0.0.0",
                port=args.port,
                threaded=True,
                debug=False,
                use_reloader=False,
            )

    except Exception as e:
        print("Error", e)
        raise


if __name__ == "__main__":
    main()
