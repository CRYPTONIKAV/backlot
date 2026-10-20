"""Notion API (read-only).

Base URL for a client: ``http://<host>/notion/v1/`` (the notion-client SDK appends ``/v1/`` to
its ``base_url``, so point it at ``http://<host>/notion``). Bearer auth
(``Authorization: Bearer <token>``, that scheme and no other); the admin/service token sees
everything, a user token is ACL-filtered. Errors use Notion's envelope:
``{"object":"error","status","code","message","request_id"}``, the last of which is also the
response's ``x-notion-request-id`` (see :func:`_error`). The OAuth client endpoints Notion puts
behind Basic auth refuse in their own shape, ``{"error","request_id"}`` (see ``unmatched_path``).

**Version-aware databases.** Notion moved database querying to the *data sources* model in
``2025-09-03``. This router keys off the ``Notion-Version`` request header:

- ``2025-09-03`` and later: ``databases.retrieve`` returns a ``data_sources: [{id,name}]`` array
  and rows are read via ``POST /data_sources/{id}/query``.
- before it (2022-06-28 and the four versions older than that): ``databases.retrieve`` returns
  ``properties`` (schema) inline and rows are read via ``POST /databases/{id}/query``.

One query path per version, as on the real API: the other one answers ``invalid_request_url``,
and so does ``GET /data_sources/{id}`` under a version older than the split, which mounts no data
source route at all (see ``_unmounted_here``). ``databases.retrieve`` stays mounted and answers
in whichever of the two shapes the caller's version reads.

Backlot has one data source per database, its id assigned at import alongside the database's own.

**The header is required** on every route here, as Notion requires it on every REST request: a
request that omits it -- or names a version Notion does not publish -- is answered
``missing_version``, behind both 401s a request with no usable credential gets -- the one that
names the bearer format and the one that says the token does not resolve (see ``_refusal``) --
and behind the 400 a URL Notion does not publish gets ahead of everything (see
``unmatched_path``). Every route declares it in the spec as well, so a client generated from that
spec -- ``backlot mcp``'s tools among them -- can send what the route asks for.

Object mapping: a Notion *page* is one doc (``subtype='page'``); a *database* is one doc
(``subtype='database'``, ``content`` → its description); a *database row* is a page whose
``parent`` is the database. Page ``content`` is served verbatim as the block tree
(``blocks.children.list``); the join of the blocks' plain_text reconstructs ``content`` exactly.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

from backlot import auth, pagination, published, store, synth
from backlot.openapi import qp

router = APIRouter(prefix="/notion/v1", tags=["notion"])
# Everything under `/notion` that no route above answers. Mounted after `router` in
# `backlot.main`, so a real route wins and only what is left reaches this one.
unmatched_router = APIRouter(prefix="/notion", include_in_schema=False)

_PAGE_MAX = 100  # Notion caps page_size at 100
# The version that split databases into data sources, so the cut between the two models: a
# request at or after it reads rows through a data source, one before it through the database.
# Notion names a version for its release date and lists them in that order, which is what makes
# ``_data_sources_model`` a date comparison ("Changes by version", read 2026-09-15).
DATA_SOURCES_VERSION = "2025-09-03"
# Every version Notion publishes, oldest first. Not a list off the document -- "Changes by
# version" omits 2021-05-11 -- but the enumeration the API itself answers an unrecognised version
# with, read off that refusal on 2026-09-17.
PUBLISHED_VERSIONS = (
    "2021-05-11",
    "2021-05-13",
    "2021-08-16",
    "2022-02-22",
    "2022-06-28",
    "2025-09-03",
    "2026-03-11",
)


# --- OpenAPI enrichment --------------------------------------------------
# GET query params are documented with openapi_extra (merges with path params, no signature
# change); POST bodies (search/query) are read via _body_object, so their shape is documented as
# a requestBody the same way. Response models use extra="allow" to preserve the full field set.
# Error paths return JSONResponse (_error), which FastAPI passes through unfiltered.


class _NLoose(BaseModel):
    model_config = ConfigDict(extra="allow")


class NotionObject(_NLoose):
    object: str
    id: str


class NotionList(_NLoose):
    object: str
    results: list[dict] = []
    has_more: bool = False


_P_PAGINATE = [qp("start_cursor"), qp("page_size", "integer")]
_P_FILTER_PROPERTIES = qp(
    "filter_properties",
    description=(
        "A property id or name; repeat the parameter, or spell it `filter_properties[]` or "
        "`filter_properties[0]`, to list several. A page inside a database comes back with only "
        "the listed properties (none when the value is unknown or empty); a page outside a "
        "database keeps its title whatever is listed."
    ),
)
_P_COMMENTS = [qp("block_id", required=True), *_P_PAGINATE]


def _body(props: dict) -> dict:
    return {
        "requestBody": {
            "content": {"application/json": {"schema": {"type": "object", "properties": props}}}
        }
    }


_B_SEARCH = _body(
    {
        "query": {"type": "string"},
        "filter": {"type": "object"},
        "sort": {"type": "object"},
        "start_cursor": {"type": "string"},
        "page_size": {"type": "integer"},
    }
)


def _version_param(description: str) -> dict:
    """The ``Notion-Version`` header, declared on every operation and required, the way the
    vendor's OpenAPI declares it (all 64 of its operations point at one ``components/parameters``
    entry; read 2026-09-16). Here it is the contract each route enforces (see ``_refusal``), and
    it is the whole of what a generated client knows -- ``backlot mcp`` builds its tools off this
    spec and its bridge sends no version of its own.

    Declared route by route because the two query routes name the versions they serve (see
    ``_query_extra``). The schema is the seven versions the route accepts, not the vendor's enum
    of its current version alone."""
    return {
        "name": "Notion-Version",
        "in": "header",
        "required": True,
        "schema": {"type": "string", "enum": list(PUBLISHED_VERSIONS)},
        "description": description,
    }


_P_VERSION = _version_param(
    "The API version to read this request under. Notion requires it on every request, and refuses "
    "a request that omits it or names a version it does not publish. `2025-09-03` and later read "
    "a database's rows through "
    "`data_sources/{id}/query`; the versions before it read them through `databases/{id}/query`, "
    "and see a database's schema inline instead of a `data_sources` array."
)


def _params(*query_params: dict) -> dict:
    """``openapi_extra`` for a route: the version header every request carries, then whatever
    query parameters the route reads."""
    return {"parameters": [_P_VERSION, *query_params]}


def _query_extra(serves: str, refuses: str) -> dict:
    """``openapi_extra`` for one of the two query routes. ``serves`` names the versions it is the
    path for, ``refuses`` the ones that read rows through the other."""
    return {
        **_body({"start_cursor": {"type": "string"}, "page_size": {"type": "integer"}}),
        "parameters": [
            _version_param(
                f"The API version to read this request under. This path serves {serves}; "
                f"{refuses} read a database's rows through the other query path, and a request "
                "under one of them is refused here, as is a request that sends no version."
            )
        ],
    }


_B_QUERY_DATA_SOURCE = _query_extra("2025-09-03 and later", "versions before it")
_B_QUERY_DATABASE = _query_extra("versions up to and including 2022-06-28", "2025-09-03 and later")


# --------------------------------------------------------------------------- helpers


_BEARER_REQUIRED = 'Authorization header must use the format "Bearer <token>".'
_TOKEN_INVALID = "API token is invalid."


def _request_id(request: Request) -> str:
    """The id every Notion answer names itself by, in the body and on the response.

    Measured 2026-09-22 with no credential, which is enough because the URL and the credential are
    checked before anything else: every refusal carries `"request_id": "<uuid>"` and
    `x-notion-request-id` with that same value. Real's is per response — three calls to one URL
    answered three ids — and this one is derived from the method, path and query string instead,
    the choice this repository makes for a synthesised id so that a corpus served twice answers the
    same thing and a test can assert one. Two POSTs that differ only in their body share an id
    here, where real would answer two; the body is left out because a request id is read from a
    log line beside a URL, not recomputed from what was sent.
    """
    raw = hashlib.shake_256(
        f"notion-req:{request.method} {request.url.path}?{request.url.query}".encode()
    )
    return str(uuid.UUID(bytes=raw.digest(16), version=4))


def _error(
    request: Request, status: int, code: str, message: str, *, extra: dict | None = None
) -> JSONResponse:
    request_id = _request_id(request)
    content = {
        "object": "error",
        "status": status,
        "code": code,
        "message": message,
        "request_id": request_id,
    }
    if extra:
        content.update(extra)
    return JSONResponse(
        status_code=status,
        content=content,
        headers={"x-notion-request-id": request_id},
    )


def _bearer_credential(request: Request) -> str | None:
    """The token from `Authorization: Bearer <t>`, and from nothing else.

    `auth.bearer_token` also takes GitHub's legacy `token <t>`, which Notion does not: measured
    2026-09-22, `token <anything>`, `Basic …`, a bare value with no scheme, a `Bearer` with no
    token and no header at all are each refused as a header that is not a bearer credential, where
    `Bearer <a token that does not resolve>` is refused as a token. The header is exactly two
    words: on 2026-09-28 `Bearer a b`, `Bearer a  b` and `Bearer nope x` were told the format, and
    `Bearer  nope` with two spaces, `Bearer` and `nope` with a tab between them, `bearer nope` and
    `BEARER nope` were told the token: two spaces or a tab separate the words as one space does,
    and the scheme is read without its case.
    """
    parts = (request.headers.get("authorization") or "").split()
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1]
    return None


def _version(request: Request) -> str:
    """The caller's ``Notion-Version``. The fallback is unreachable through a route -- every one
    of them runs ``_refusal`` first, which answers a request that sent no version -- and is here
    so a reader of ``_database_obj`` or ``search`` need not prove that to know what the version is
    when the shape is built."""
    return request.headers.get("notion-version") or DATA_SOURCES_VERSION


def _data_sources_model(version: str) -> bool:
    """Whether ``version`` reads a database's rows through a data source.

    2025-09-03 re-organised the ``/v1/databases`` APIs into ``/v1/data_sources`` (for a data
    source) and ``/v1/databases`` (for the container that holds them). Every version before it
    predates data sources entirely, so the comparison is against the date, not against the one
    legacy version a client happens to send: 2022-02-22 has no more idea what a data source is
    than 2022-06-28 does.

    A value Notion does not publish never reaches this comparison -- ``_refusal`` answers it
    ``missing_version``, as api.notion.com does (measured 2026-09-17), so the ordering here is
    total over the seven versions Notion publishes."""
    return version >= DATA_SOURCES_VERSION


def _unknown_version(value: str) -> str:
    """Notion's refusal for a version it does not publish: the enumeration, then the value it was
    handed, both quoted the way the live API quotes them (measured 2026-09-17)."""
    listed = ", ".join(f'`"{v}"`' for v in PUBLISHED_VERSIONS[:-1])
    return (
        f"Notion-Version header failed validation: Notion-Version header should be {listed}, "
        f'or `"{PUBLISHED_VERSIONS[-1]}"`, instead was `"{value}"`.'
    )


def _refusal(request: Request, caller) -> JSONResponse | None:
    """What a request is answered before any route reads a row -- None when it may go on. Every
    route calls this, so the two checks are in one place and in the order the real API applies
    them. Called rather than declared as a router dependency: the credential half has to stay
    where the route resolves the caller it goes on to scope rows by.

    The credential first: on 2026-09-15 an invalid token answered ``unauthorized`` on both query
    paths under 2022-06-28, under 2025-09-03 and with no version header at all, so the version is
    never what a request without a usable credential hears about. Which of that 401's two messages
    it gets is :func:`_bearer_credential`: a header that is not a bearer credential is told the
    format, and a bearer the server cannot resolve is told the token.

    Then the version, which Notion requires on every request and refuses when it is one Notion
    does not publish. Both halves answer ``missing_version``, and each has its own message:
    measured against api.notion.com on 2026-09-17 with an integration token, a version-less
    request is refused on twelve routes out of twelve, and ``banana``, ``2024-01-01`` and
    ``2099-01-01`` are each answered with the enumeration of the seven versions it does publish.
    The absent-header message is a pair of backticks wider than the example Notion's status-code
    table prints for the code, so the table is stale against the service there.

    An empty header value is a value rather than a missing one, and real answers it the
    enumerating way -- so it is refused here too, which is what keeps it from sorting below
    2025-09-03 and quietly picking the legacy model for a caller that named nothing."""
    if _bearer_credential(request) is None:
        return _error(request, 401, "unauthorized", _BEARER_REQUIRED)
    if caller is None:
        return _error(request, 401, "unauthorized", _TOKEN_INVALID)
    version = request.headers.get("notion-version")
    if version is None:
        return _error(
            request,
            400,
            "missing_version",
            "Notion-Version header failed validation: Notion-Version header should be defined, "
            "instead was `undefined`.",
        )
    if version not in PUBLISHED_VERSIONS:
        return _error(request, 400, "missing_version", _unknown_version(version))
    return None


def _unmounted_here(request: Request, *, data_sources: bool) -> JSONResponse | None:
    """Notion's answer when the route the caller asked for is not mounted under the version it
    sent -- None when it is. Runs after ``_refusal``, so the version read here is one the caller
    sent.

    The two query paths, measured against api.notion.com on 2026-09-11 with an integration token,
    probing each with an id that exists but is a page: 2022-06-28 serves ``databases/{id}/query``
    and answers ``invalid_request_url`` for ``data_sources/{id}/query``, and 2025-09-03 the other
    way round. ``GET /data_sources/{id}`` is gated on the version the same way, measured on
    2026-09-17 with an id that exists and is not a data source: 2021-05-11 and 2022-06-28 answer
    ``invalid_request_url``, 2025-09-03 and 2026-03-11 answer ``object_not_found``, which is the
    route answering. Its counterpart ``GET /databases/{id}`` answered under each version probed
    there -- 2021-05-11, 2022-06-28 and 2026-03-11.

    The 09-11 probe recorded codes rather than bodies, and reported the refusal as "the same as a
    made-up path", so the message here is what api.notion.com answered on 2026-09-15 for a path no
    version mounts at all."""
    if _data_sources_model(_version(request)) is not data_sources:
        return _error(request, 400, "invalid_request_url", "Invalid request URL.")
    return None


def _norm(nid: str) -> str:
    """Notion accepts an id dashed or dashless, and does not fold its case: a dashed id in upper
    case was a 404 on ``pages/{id}`` (measured 2026-09-30) and on ``comments`` (2026-10-02), where
    a dashless one in upper case was not read as a uuid at all. Folding case here serves an id
    real does not. Canonicalize to the dashed lowercase form the stored ``id`` /
    ``data_source_id`` columns actually hold (see ``synth._uuid_from``) rather than to a dashless
    key: a UUID's dashes sit at fixed offsets (8-4-4-4-12), so reconstructing them is
    deterministic, and the column's job is to hold the value the API reports, not a lookup key a
    reader has to rebuild dashes from.

    Malformed input (the wrong length once dashes are stripped) comes back unchanged, which
    matches no stored id. Real refuses a malformed id with a 400 before it looks anything up;
    ``list_comments`` checks its ``block_id`` that way first, and the path routes answer it the
    404 an unknown id gets."""
    h = (nid or "").replace("-", "").lower()
    if len(h) != 32:
        return h
    return f"{h[0:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"


def _existing_id(request: Request, page_id: str) -> str | None:
    """The canonical form of a served page/database/block id, or None if it names nothing -- a
    PRIMARY KEY lookup (see store.notion_by_id). All it does is normalise the spelling and
    confirm a row holds that id.

    Unscoped by ACL on purpose: used only by _query_rows, for the ``databases/{id}/query`` half,
    which needs the id to drive a further, separately ACL-scoped query (store.children, a
    DIFFERENT table) rather than to serve this row itself -- see get_page and friends (and
    list_comments), which read the row directly instead of going through here, so as not to
    resolve it once and refetch it a second time for the ACL check."""
    row = store.notion_by_id(auth.conn(request), _norm(page_id))
    return row["id"] if row is not None else None


def _db_doc_for_data_source(request: Request, dsid: str) -> str | None:
    """The database `id` behind a data source id -- a unique-indexed column lookup (see
    store.notion_by_data_source_id), not the O(pages) scan -- hashing every notion row's data
    source id on every request. Unscoped by ACL for the same reason as
    _existing_id."""
    row = store.notion_by_data_source_id(auth.conn(request), _norm(dsid))
    return row["id"] if row is not None else None


# Request members real Notion refuses, route by route. Measured against api.notion.com on
# 2026-09-30 under Notion-Version 2025-09-03 and 2022-06-28: every operation validates the query
# string and the body against its own schema and answers 400 `validation_error` naming the member
# it refused, apart from `GET /v1/pages/{id}`, which serves an unknown query parameter (measured
# with `foo=1`) and is therefore left out of the sweep. The allowlists are per route rather than
# per router: a `page_size` in the query string of `POST /v1/search` is not that route's member
# (its pagination sits in the body), and real refuses it as an extra key. The two query routes
# take no query member either: that half is an extrapolation from the search row, applied so a
# misspelled key is refused there the same way instead of being served.

_MISSING = object()
_NEGATIVE_PAGE = object()

_Q_PAGINATE = frozenset({"start_cursor", "page_size"})
_Q_COMMENTS = frozenset({"block_id", "start_cursor", "page_size"})
_Q_SEARCH = frozenset({"filter_properties"})
_Q_NONE: frozenset[str] = frozenset()  # both query routes: no query member at all

_B_SEARCH_MEMBERS = frozenset({"query", "filter", "sort", "start_cursor", "page_size"})
_B_QUERY_MEMBERS = frozenset({"filter", "sorts", "start_cursor", "page_size"})

# The reserved prefix of the cursor this router hands back on the routes whose real cursor is a
# uuid, and the shape real reads as one (the dashed 8-4-4-4-12 form in either case, or 32 lower
# case hex digits, the same pair list_comments reads for its block id).
_OWN_CURSOR_RE = re.compile(r"00000000-0000-4000-8000-([0-9a-f]{12})")
_UUID_RE = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
    r"|[0-9a-f]{32}"
)
_PAGE_SIZE_RE = re.compile(r"-?[0-9]+")


def _inline(value) -> str:
    """``value`` the way a Notion refusal quotes it back: compact JSON, `undefined` for a member
    the request did not send."""
    if value is _MISSING:
        return "undefined"
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _query_inline(raw: str) -> str:
    """A query string's value in that same quoting. The query string carries text, and real reads
    `1` back as a number and `abc` as a string (measured 2026-09-30)."""
    try:
        return _inline(json.loads(raw))
    except ValueError:
        return _inline(raw)


def _unknown_query(
    request: Request, allowed: frozenset[str], *, brackets: frozenset[str] = frozenset()
) -> JSONResponse | None:
    """The 400 for the first query parameter the route does not take, or None when every key
    belongs to it.

    ``brackets`` names the allowed keys whose bracketed spellings belong to them too; only
    ``filter_properties`` was measured to work that way (`[]`, `[0]`, ...), and its own reader
    judges those. Every other allowed key matches exactly, so `start_cursor[0]` or `page_size[0]`
    is a member the route does not take. Real's message names the key and quotes the value it
    read."""
    for key, value in request.query_params.multi_items():
        if key in allowed or (key.split("[", 1)[0] in brackets and "[" in key):
            continue
        return _error(
            request,
            400,
            "validation_error",
            f"query failed validation: query.{key} should be not present, "
            f"instead was `{_query_inline(value)}`.",
        )
    return None


def _unknown_body(request: Request, body: dict, allowed: frozenset[str]) -> JSONResponse | None:
    """The 400 for the first body key the route does not take (see :func:`_unknown_query`)."""
    for key, value in body.items():
        if key in allowed:
            continue
        return _error(
            request,
            400,
            "validation_error",
            f"body failed validation: body.{key} should be not present, "
            f"instead was `{_inline(value)}`.",
        )
    return None


def _page_refusal(request: Request, where: str, message: str) -> JSONResponse:
    return _error(request, 400, "validation_error", f"{where} failed validation: {message}")


def _page_range(n: int, family: str, request: Request, where: str):
    """Apply the family's range rule to an integer page size.

    ``comments`` refuses every value outside 1..100. The list routes read `0` and anything over
    the cap as a full page, answer a negative with an empty page carrying the first item's own
    cursor (real does that for `-1` on all four, measured), and `users` answers -2 and below with
    the 503 its public list endpoint returns for those three values."""
    if family == "comments":
        if n < 1:
            return None, _page_refusal(
                request,
                where,
                f"{where}.page_size should be \u2265 `1` or `undefined`, instead was `{n}`.",
            )
        if n > _PAGE_MAX:
            return None, _page_refusal(
                request,
                where,
                f"{where}.page_size should be \u2264 `100` or `undefined`, instead was `{n}`.",
            )
        return n, None
    if family == "users" and n <= -2:
        return None, _error(
            request,
            503,
            "service_unavailable",
            "Public API service is temporarily unavailable, please try again later.",
            extra={"additional_data": {"endpoint_name": "public_listUsers"}},
        )
    if n < 0:
        return _NEGATIVE_PAGE, None
    if n == 0 or n > _PAGE_MAX:
        return _PAGE_MAX, None
    return n, None


def _query_page_size(request: Request, family: str):
    """``(page_size, refusal)`` for a route that reads `page_size` from the query string.

    A repeated parameter is refused whatever it repeats -- `1&1` included -- with real's
    "Conflicting values" wording (measured 2026-09-30); an empty value is the default page.
    Anything that is not an integer -- `abc`, `1.5`, `1_0` -- is the number's 400, with the value
    quoted the way real quotes it."""
    raw = request.query_params.getlist("page_size")
    if len(raw) > 1:
        return None, _error(
            request,
            400,
            "validation_error",
            'Conflicting values for "page_size" in query parameters.',
        )
    value = raw[-1] if raw else None
    if value is None or not value.strip():
        return _PAGE_MAX, None
    if not _PAGE_SIZE_RE.fullmatch(value):
        return None, _page_refusal(
            request,
            "query",
            f"query.page_size should be a number or `undefined`, "
            f"instead was `{_query_inline(value)}`.",
        )
    return _page_range(int(value), family, request, "query")


def _body_page_size(body: dict, family: str, request: Request):
    """``(page_size, refusal)`` for one of the two POST bodies. A member the body did not send is
    the default page; a bool is not a number, though Python would read it as one."""
    if "page_size" not in body:
        return _PAGE_MAX, None
    value = body["page_size"]
    if isinstance(value, bool) or not isinstance(value, int):
        return None, _page_refusal(
            request,
            "body",
            f"body.page_size should be a number or `undefined`, instead was `{_inline(value)}`.",
        )
    return _page_range(value, family, request, "body")


def _uuid_cursor(offset: int) -> str:
    """The next-page cursor of the routes whose real cursor is a uuid (children, comments and
    search): the offset in the last twelve hex digits of a reserved uuid.

    Real's own refusal on those routes names the shape ("should be a valid uuid"), so the token
    this server hands back takes it too, and the zero uuid real refuses stays outside the reserved
    prefix."""
    return f"00000000-0000-4000-8000-{offset:012x}"


def _own_cursor_offset(token: str) -> int | None:
    m = _OWN_CURSOR_RE.fullmatch(token)
    return int(m[1], 16) if m else None


def _position_after(token: str, ids: list[str]) -> int | None:
    """Where a cursor that names an item resumes: the item after it, or None when it names
    nothing the route would serve."""
    wanted = _norm(token)
    for i, served in enumerate(ids):
        if _norm(served) == wanted:
            return i + 1
    return None


def _cursor(token, ids: list[str], family: str, request: Request, where: str):
    """``(offset, refusal)`` for a `start_cursor`, in the two families real validates.

    ``users`` and the data-source query take an opaque cursor: theirs is refused with "The
    start_cursor provided is invalid: <value>" for anything that is not one, the zero uuid
    included. Children, comments and search name the uuid shape instead and quote the value (the
    three wordings the measurement records). A cursor this server emitted is the page it names; a
    token that names an item resumes after it. An empty value is refused too -- the user list says
    it is not a string, with the `0` real reads out of it."""
    if token is None:
        return 0, None
    if not isinstance(token, str):
        # A body member of the wrong type is refused before anything decodes it (the query string
        # can only carry text, so this is the POST half).
        if family in ("users", "query"):
            return None, _error(
                request,
                400,
                "validation_error",
                f"The start_cursor provided is invalid: {_inline(token)}",
            )
        return None, _error(
            request,
            400,
            "validation_error",
            f"{where} failed validation: {where}.start_cursor should be a valid uuid or "
            f"`undefined`, instead was `{_inline(token)}`.",
        )
    if family in ("users", "query"):
        if not token:
            if family == "users" and where == "query":
                return None, _page_refusal(
                    request,
                    "query",
                    "query.start_cursor should be a string or `undefined`, instead was `0`.",
                )
            return None, _error(
                request,
                400,
                "validation_error",
                f"The start_cursor provided is invalid: {_inline(token)}",
            )
        offset = pagination.decode_cursor_or_none(token)
        if offset is not None:
            return offset, None
        if family == "query" and token.startswith("s:"):
            parts = token[2:].split(":")
            if len(parts) == 2:
                resume = _position_after(parts[0], ids)
                if resume is not None:
                    return resume, None
        resume = _position_after(token, ids)
        if resume is not None:
            return resume, None
        return None, _error(
            request,
            400,
            "validation_error",
            f"The start_cursor provided is invalid: {token}",
        )
    offset = _own_cursor_offset(token)
    if offset is not None:
        return offset, None
    if _UUID_RE.fullmatch(token):
        resume = _position_after(token, ids)
        if resume is not None:
            return resume, None
        return None, _error(
            request,
            400,
            "validation_error",
            f"The start_cursor provided is invalid: {token}",
        )
    return None, _error(
        request,
        400,
        "validation_error",
        f"{where} failed validation: {where}.start_cursor should be a valid uuid or `undefined`, "
        f"instead was `{_inline(token)}`.",
    )


async def _body_object(request: Request) -> tuple[dict, JSONResponse | None]:
    """The request's JSON object, or the 400 real answers for a body it cannot read.

    A body that parses as a JSON object is judged by its members whatever content type carries
    it: an empty body, no content type and `text/plain` with `{}` are a 200 on both sides
    (measured 2026-09-30), and reading the object behind a `text/plain` body is the same rule --
    a body whose members are visible is validated, never silently dropped. Everything that
    declares `application/json` is parsed strictly: invalid JSON is `invalid_json` with "Error
    parsing JSON body.", and a parsed value that is not an object gets the object's 400, `[]`
    quoted back the way real does. A non-JSON content type carrying neither an object nor JSON is
    read as no members."""
    content_type = (request.headers.get("content-type") or "").split(";")[0].strip().lower()
    declared_json = not content_type or content_type == "application/json"
    raw = await request.body()
    if not raw.strip():
        return {}, None
    try:
        parsed = json.loads(raw)
    except ValueError:
        if not declared_json:
            return {}, None
        return {}, _error(request, 400, "invalid_json", "Error parsing JSON body.")
    if isinstance(parsed, dict):
        return parsed, None
    if not declared_json:
        return {}, None
    return {}, _error(
        request,
        400,
        "validation_error",
        f"body failed validation: body should be an object, instead was `{_inline(parsed)}`.",
    )


def _search_filter_refusal(request: Request, body: dict, version: str) -> JSONResponse | None:
    """The 400 for a search `filter` real refuses, or None for one it serves.

    Real's rule is a union: `property` must be `"object"` and `value` one of the version's object
    types, or `in_trash` must be defined. When neither branch holds the message lists the line
    each branch failed on, which is why the measured refusals carry the `in_trash` line after the
    one that is actually wrong. Under 2022-06-28 `database` is the database row's type; from
    2025-09-03 the same row is a `data_source` (measured 2026-09-30 and 2026-10-14)."""
    flt = body["filter"]
    allowed = (
        ("page", "data_source")
        if _data_sources_model(version)
        else ("page", "database", "data_source")
    )
    lines = []
    prop = flt.get("property", _MISSING)
    if prop != "object":
        lines.append(f'body.filter.property should be `"object"`, instead was `{_inline(prop)}`.')
    value = flt.get("value", _MISSING)
    if value not in allowed:
        listed = " or ".join(f'`"{v}"`' for v in allowed)
        lines.append(f"body.filter.value should be {listed}, instead was `{_inline(value)}`.")
    if not lines:
        return None
    lines.append("body.filter.in_trash should be defined, instead was `undefined`.")
    return _error(
        request,
        400,
        "validation_error",
        "body failed validation. Fix one:\n" + "\n".join(lines),
    )


def _search_sort_refusal(request: Request, body: dict) -> JSONResponse | None:
    """The 400 for a search `sort` real refuses, or None for one it serves.

    Real sorts either by relevance, naming the property (`property` must be `"relevance"`), or by
    a timestamp, where `timestamp` must be `last_edited_time` and `property` must be absent. The
    direction's refusal is the mixed-branch message the measurement records, with the timestamp
    branch's own line beneath it."""
    sort = body["sort"]
    if not isinstance(sort, dict):
        return _error(
            request,
            400,
            "validation_error",
            f"body failed validation: body.sort should be an object or `undefined`, "
            f"instead was `{_inline(sort)}`.",
        )
    direction = sort.get("direction", _MISSING)
    direction_ok = direction is _MISSING or direction in ("ascending", "descending")
    if "property" in sort:
        if sort["property"] != "relevance":
            return _error(
                request,
                400,
                "validation_error",
                f'body.sort.property should be "relevance", instead was `{sort["property"]}`.',
            )
        if direction_ok:
            return None
        return _error(
            request,
            400,
            "validation_error",
            'body failed validation: body.sort.direction should be `"ascending"`, '
            f'`"descending"`, or `undefined`, instead was `{_inline(direction)}`.',
        )
    if sort.get("timestamp", _MISSING) != "last_edited_time":
        return _error(
            request,
            400,
            "validation_error",
            'body.sort.timestamp should be "last_edited_time" when sorting by timestamp.',
        )
    if not direction_ok:
        return _error(
            request,
            400,
            "validation_error",
            "body failed validation. Fix one:\n"
            'body.sort.direction should be `"ascending"`, `"descending"`, or `undefined`, '
            f"instead was `{_inline(direction)}`.\n"
            "body.sort.property should be defined, instead was `undefined`.",
        )
    return None


def _query_body_refusal(request: Request, body: dict, db) -> JSONResponse | None:
    """The 400 for a data-source query body real refuses, or None for one it serves.

    A filter or a sort that names a property the database does not have is refused by name
    (measured 2026-09-30); a member of the wrong type is refused the way the rest of the router
    refuses one."""
    names = set(_schema_props(db))
    if "filter" in body:
        flt = body["filter"]
        if not isinstance(flt, dict):
            return _error(
                request,
                400,
                "validation_error",
                f"body failed validation: body.filter should be an object or `undefined`, "
                f"instead was `{_inline(flt)}`.",
            )
        if "property" in flt and str(flt["property"]) not in names:
            return _error(
                request,
                400,
                "validation_error",
                f"Could not find property with name or id: {flt['property']}",
            )
    if "sorts" in body:
        sorts = body["sorts"]
        if not isinstance(sorts, list):
            return _error(
                request,
                400,
                "validation_error",
                f"body failed validation: body.sorts should be an array or `undefined`, "
                f"instead was `{_inline(sorts)}`.",
            )
        for sort in sorts:
            if not isinstance(sort, dict):
                return _error(
                    request,
                    400,
                    "validation_error",
                    "body failed validation: body.sorts should be an array of objects, "
                    f"instead was `{_inline(sort)}`.",
                )
            if "property" in sort and str(sort["property"]) not in names:
                return _error(
                    request,
                    400,
                    "validation_error",
                    f"Could not find sort property with name or id: {sort['property']}",
                )
    return None


def _first_cursor_page(cursor: str | None, type_key: str) -> dict:
    """The page real answers a negative `page_size` with: no results, `has_more` true, and the
    first item's own cursor (the first user's, block's or page's id; an `s:` token on the query)."""
    return {
        "object": "list",
        "results": [],
        "next_cursor": cursor,
        "has_more": cursor is not None,
        "type": type_key,
        type_key: {},
    }


def _list_obj(
    results: list,
    offset: int,
    page_len: int,
    total: int,
    type_key: str,
    *,
    encode=pagination.encode_cursor,
) -> dict:
    nxt = offset + page_len
    has_more = nxt < total
    return {
        "object": "list",
        "results": results,
        "next_cursor": encode(nxt) if has_more else None,
        "has_more": has_more,
        "type": type_key,
        type_key: {},
    }


def _user_obj(conn, email: str) -> dict:
    u = store.get_user(conn, email)
    name = u["display_name"] if u else (email.split("@")[0] if email else "Unknown")
    return {
        "object": "user",
        "id": synth.notion_user_id(email or ""),
        "type": "person",
        "name": name,
        "avatar_url": None,
        # Real's person member is `email` then `email_verified` under each Notion-Version, the
        # latter `true` for the one person measured (2026-10-13). The corpus records no
        # verification, so every person is served `true`.
        "person": {"email": email, "email_verified": True},
    }


def _emoji_icon(icon: str | None) -> dict | None:
    if not icon:
        return None
    if icon.startswith("http"):
        return {"type": "external", "external": {"url": icon}}
    return {"type": "emoji", "emoji": icon}


def _cover(url: str | None) -> dict | None:
    return {"type": "external", "external": {"url": url}} if url else None


def _parent_field(row) -> dict:
    # `parent_id` already HOLDS the parent's served id — the importer resolved it there.
    # Hashing it again would name a page nothing serves.
    if row["parent_id"]:
        return {"type": "database_id", "database_id": row["parent_id"]}
    return {"type": "workspace", "workspace": True}


def _title_prop(title: str) -> dict:
    return {"id": "title", "type": "title", "title": synth.notion_rich_text(title)}


def _prop_value(name: str, value) -> dict:
    """A Notion property *value* object (on a database row)."""
    if isinstance(value, bool):
        return {"id": name, "type": "checkbox", "checkbox": value}
    if isinstance(value, (int, float)):
        return {"id": name, "type": "number", "number": value}
    if isinstance(value, str):
        return {"id": name, "type": "select", "select": {"name": value, "color": "default"}}
    return {"id": name, "type": "rich_text", "rich_text": synth.notion_rich_text(str(value))}


def _page_obj(conn, row) -> dict:
    pid = row["id"]
    created = row["created_ts"]
    updated = row["updated_ts"] or created
    props = {"title": _title_prop(row["title"])}
    for k, v in (json.loads(row["properties"]) if row["properties"] else {}).items():
        props[k] = _prop_value(k, v)
    return {
        "object": "page",
        "id": pid,
        "created_time": synth.rfc3339(created),
        "last_edited_time": synth.rfc3339(updated),
        "created_by": _user_obj(conn, row["author_email"]),
        "last_edited_by": _user_obj(conn, row["author_email"]),
        "cover": _cover(row["cover"]),
        "icon": _emoji_icon(row["icon"]),
        "parent": _parent_field(row),
        "archived": False,
        "in_trash": False,
        "properties": props,
        "url": f"https://www.notion.so/{pid.replace('-', '')}",
        "public_url": None,
    }


_FILTER_PROPERTIES_KEY = re.compile(r"filter_properties((?:\[[^\[\]]*\])*)")
_BRACKET = re.compile(r"\[([^\[\]]*)\]")


def _filter_properties_keep(request: Request) -> list[str] | None | JSONResponse:
    """The ``filter_properties`` names, ``None`` when the parameter is absent, or the 400 real
    answers. Measured against api.notion.com (2022-06-28 and 2025-09-03) on 2026-10-15, which reads
    the query string the way Node's ``qs`` does: the bare key, ``[]`` and an index from 0 to 20
    all add to one list; any other single bracket (``[x]``, ``[21]``, ``[01]``) is a 400 unless the
    bare key came first, and then lists nothing; a deeper bracket (``[0][x]``) or an index given
    twice lists nothing. A bare ``[][]=Tags`` kept ``Tags`` on real and keeps nothing here."""
    keep: list[str] = []
    slots: dict[str, list[str]] = {}
    first = refused = None
    for key, value in request.query_params.multi_items():
        if (m := _FILTER_PROPERTIES_KEY.fullmatch(key)) is None:
            continue
        brackets = _BRACKET.findall(m[1])
        first = brackets if first is None else first
        if brackets in ([], [""]):
            keep.append(value)
        elif len(brackets) > 1:
            continue
        elif (i := brackets[0]).isascii() and i.isdigit() and str(int(i)) == i and int(i) <= 20:
            slots.setdefault(i, []).append(value)
        else:
            refused = first != []
    if first is None:
        return None
    if refused:
        return _error(
            request,
            400,
            "validation_error",
            'The value of "filter_properties" in query parameters must be a string or list of '
            "strings",
        )
    return keep + [values[0] for values in slots.values() if len(values) == 1]


def _filter_properties(page: dict, keep: list[str] | None) -> dict:
    """Narrow ``properties`` to those whose id or name is in ``keep``; ``None`` leaves the page
    as it is."""
    if keep is None:
        return page
    wanted = set(keep)
    page["properties"] = {
        k: v for k, v in page["properties"].items() if k in wanted or v["id"] in wanted
    }
    return page


def _schema_props(row) -> dict:
    """A Notion database property *schema* (on a database / data source)."""
    schema = json.loads(row["properties"]) if row["properties"] else {}
    out = {"title": {"id": "title", "name": "Name", "type": "title", "title": {}}}
    for k, v in schema.items():
        t = v.get("type", "rich_text") if isinstance(v, dict) else "rich_text"
        out[k] = {"id": k, "name": k, "type": t, t: {}}
    return out


def _database_obj(conn, row, version: str) -> dict:
    did = row["id"]
    created = row["created_ts"]
    updated = row["updated_ts"] or created
    obj = {
        "object": "database",
        "id": did,
        "created_time": synth.rfc3339(created),
        "last_edited_time": synth.rfc3339(updated),
        "created_by": _user_obj(conn, row["author_email"]),
        "last_edited_by": _user_obj(conn, row["author_email"]),
        "title": synth.notion_rich_text(row["title"]),
        "description": synth.notion_rich_text(row["content"]),
        "icon": _emoji_icon(row["icon"]),
        "cover": _cover(row["cover"]),
        "parent": {"type": "workspace", "workspace": True},
        "archived": False,
        "in_trash": False,
        "is_inline": False,
        "url": f"https://www.notion.so/{did.replace('-', '')}",
        "public_url": None,
    }
    if _data_sources_model(version):
        obj["data_sources"] = [{"id": row["data_source_id"], "name": row["title"]}]
    else:
        obj["properties"] = _schema_props(row)
    return obj


def _data_source_obj(conn, row) -> dict:
    return {
        "object": "data_source",
        "id": row["data_source_id"],
        "name": row["title"],
        "parent": {"type": "database_id", "database_id": row["id"]},
        "database_parent": {"type": "database_id", "database_id": row["id"]},
        "properties": _schema_props(row),
    }


# --------------------------------------------------------------------------- pages / blocks


@router.get("/pages/{page_id}", response_model=NotionObject, openapi_extra=_params())
async def get_page(page_id: str, request: Request):
    caller = auth.resolve_bearer(request)
    if (refusal := _refusal(request, caller)) is not None:
        return refusal
    conn = auth.conn(request)
    # One ACL-scoped query, not a resolve followed by a get_document refetch of the same row:
    # `id` is the PRIMARY KEY, so the ACL clause can only narrow "found" to "not found", never
    # redirect to a different row (see store.notion_by_id).
    row = store.notion_by_id(conn, _norm(page_id), auth.visible_ids(request, caller))
    if row is None or row["subtype"] == "database":
        return _error(request, 404, "object_not_found", f"Could not find page with ID: {page_id}.")
    return _page_obj(conn, row)


@router.get("/blocks/{block_id}", response_model=NotionObject, openapi_extra=_params())
async def get_block(block_id: str, request: Request):
    caller = auth.resolve_bearer(request)
    if (refusal := _refusal(request, caller)) is not None:
        return refusal
    conn = auth.conn(request)
    row = store.notion_by_id(conn, _norm(block_id), auth.visible_ids(request, caller))
    if row is None:
        return _error(
            request, 404, "object_not_found", f"Could not find block with ID: {block_id}."
        )
    bid = row["id"]
    kind = "child_database" if row["subtype"] == "database" else "child_page"
    return {
        "object": "block",
        "id": bid,
        "type": kind,
        "has_children": row["subtype"] != "database",
        "archived": False,
        "in_trash": False,
        "created_time": synth.rfc3339(row["created_ts"]),
        "last_edited_time": synth.rfc3339(row["updated_ts"] or row["created_ts"]),
        "parent": _parent_field(row),
        kind: {"title": row["title"]},
    }


@router.get(
    "/blocks/{block_id}/children",
    response_model=NotionList,
    openapi_extra=_params(*_P_PAGINATE),
)
async def get_block_children(block_id: str, request: Request):
    caller = auth.resolve_bearer(request)
    if (refusal := _refusal(request, caller)) is not None:
        return refusal
    # The query string is judged before the lookup, the order the comments route keeps: a member
    # real refuses is the 400 it answers, not the 404 for a block that is also missing.
    if (refusal := _unknown_query(request, _Q_PAGINATE)) is not None:
        return refusal
    limit, refusal = _query_page_size(request, "list")
    if refusal is not None:
        return refusal
    conn = auth.conn(request)
    row = store.notion_by_id(conn, _norm(block_id), auth.visible_ids(request, caller))
    if row is None:
        return _error(
            request, 404, "object_not_found", f"Could not find block with ID: {block_id}."
        )
    # A database's rows are pages reached through the query route, not blocks: on api.notion.com
    # (2026-10-04, three databases, Notion-Version 2022-06-28 and 2025-09-03) its children are the
    # empty list, and its block reports `has_children` false.
    blocks = [] if row["subtype"] == "database" else synth.notion_blocks(row["id"], row["content"])
    ids = [b["id"] for b in blocks]
    offset, refusal = _cursor(
        request.query_params.get("start_cursor"), ids, "uuid", request, "query"
    )
    if refusal is not None:
        return refusal
    if limit is _NEGATIVE_PAGE:
        return _first_cursor_page(ids[0] if ids else None, "block")
    page = blocks[offset : offset + limit]
    return _list_obj(page, offset, len(page), len(blocks), "block", encode=_uuid_cursor)


# --------------------------------------------------------------------------- databases / data sources


@router.get("/databases/{database_id}", response_model=NotionObject, openapi_extra=_params())
async def get_database(database_id: str, request: Request):
    caller = auth.resolve_bearer(request)
    if (refusal := _refusal(request, caller)) is not None:
        return refusal
    conn = auth.conn(request)
    row = store.notion_by_id(conn, _norm(database_id), auth.visible_ids(request, caller))
    if row is None or row["subtype"] != "database":
        return _error(
            request, 404, "object_not_found", f"Could not find database with ID: {database_id}."
        )
    return _database_obj(conn, row, _version(request))


@router.get("/data_sources/{data_source_id}", response_model=NotionObject, openapi_extra=_params())
async def get_data_source(data_source_id: str, request: Request):
    caller = auth.resolve_bearer(request)
    if (refusal := _refusal(request, caller)) is not None:
        return refusal
    if (refusal := _unmounted_here(request, data_sources=True)) is not None:
        return refusal
    conn = auth.conn(request)
    # No subtype check needed here, unlike get_database's lookup by `id` (which spans both pages
    # and databases): served_data_source_id is populated only for subtype='database' rows, and the
    # importer writes it or NULL on every import (see the notion block in importer.byo._Loader.add).
    # So a match here already implies the right kind.
    row = store.notion_by_data_source_id(
        conn, _norm(data_source_id), auth.visible_ids(request, caller)
    )
    if row is None:
        return _error(
            request,
            404,
            "object_not_found",
            f"Could not find data source with ID: {data_source_id}.",
        )
    return _data_source_obj(conn, row)


async def _query_rows(request: Request, row_id: str, *, data_sources: bool):
    """The rows of one database, reached by whichever of the two query paths the caller's version
    serves. ``row_id`` is the id as spelled in the path -- a data source's under ``data_sources``,
    the database's own under ``databases`` -- and is resolved only once the credential and the
    version have both been accepted, so a refused request never runs a lookup."""
    caller = auth.resolve_bearer(request)
    if (refusal := _refusal(request, caller)) is not None:
        return refusal
    if (refusal := _unmounted_here(request, data_sources=data_sources)) is not None:
        return refusal
    if (refusal := _unknown_query(request, _Q_NONE)) is not None:
        return refusal
    db_id = (
        _db_doc_for_data_source(request, row_id) if data_sources else _existing_id(request, row_id)
    )
    conn = auth.conn(request)
    visible = auth.visible_ids(request, caller)
    db = store.get_document(conn, "notion", db_id, visible_ids=visible) if db_id else None
    if db is None or db["subtype"] != "database":
        return _error(request, 404, "object_not_found", "Could not find the requested database.")
    body, refusal = await _body_object(request)
    if refusal is not None:
        return refusal
    if (refusal := _unknown_body(request, body, _B_QUERY_MEMBERS)) is not None:
        return refusal
    if (refusal := _query_body_refusal(request, body, db)) is not None:
        return refusal
    limit, refusal = _body_page_size(body, "list", request)
    if refusal is not None:
        return refusal
    token = body.get("start_cursor")
    if limit is _NEGATIVE_PAGE:
        # The empty page names the first row, so one row is all it reads.
        first = store.children(conn, "notion", db_id, visible, limit=1, offset=0)
        cursor = f"s:{first[0]['id']}:{db['data_source_id']}" if first else None
        return _first_cursor_page(cursor, "page_or_database")
    # One row past the page tells `has_more`, so the fetch is the window the request needs -- not
    # the whole row set. Only a cursor that names a row (this route's negative-size `s:` token, or
    # a caller handing back a row id) has to find that row first, and only that request reads the
    # ids; a token the route must refuse is judged before anything is read.
    if token is not None and (not isinstance(token, str) or not token):
        offset, refusal = _cursor(token, [], "query", request, "body")
        if refusal is not None:
            return refusal
    offset = pagination.decode_cursor_or_none(token) if isinstance(token, str) and token else None
    if token is None or offset is not None:
        start = offset or 0
        rows = store.children(conn, "notion", db_id, visible, limit=limit + 1, offset=start)
        page = rows[:limit]
        results = [_page_obj(conn, r) for r in page]
        return _list_obj(results, start, len(page), start + len(rows), "page_or_database")
    ids = [r["id"] for r in store.children(conn, "notion", db_id, visible, limit=-1, offset=0)]
    offset, refusal = _cursor(token, ids, "query", request, "body")
    if refusal is not None:
        return refusal
    rows = store.children(conn, "notion", db_id, visible, limit=limit + 1, offset=offset)
    page = rows[:limit]
    results = [_page_obj(conn, r) for r in page]
    return _list_obj(results, offset, len(page), offset + len(rows), "page_or_database")


@router.post(
    "/data_sources/{data_source_id}/query",
    response_model=NotionList,
    openapi_extra=_B_QUERY_DATA_SOURCE,
)
async def query_data_source(data_source_id: str, request: Request):
    return await _query_rows(request, data_source_id, data_sources=True)


@router.post(
    "/databases/{database_id}/query", response_model=NotionList, openapi_extra=_B_QUERY_DATABASE
)
async def query_database(database_id: str, request: Request):
    return await _query_rows(request, database_id, data_sources=False)


# --------------------------------------------------------------------------- search


@router.post(
    "/search",
    response_model=NotionList,
    openapi_extra={**_B_SEARCH, **_params(_P_FILTER_PROPERTIES)},
)
async def search(request: Request):
    caller = auth.resolve_bearer(request)
    if (refusal := _refusal(request, caller)) is not None:
        return refusal
    # Answered after the credential and the version and before the body, measured 2026-10-15.
    if isinstance(keep := _filter_properties_keep(request), JSONResponse):
        return keep
    if (
        refusal := _unknown_query(request, _Q_SEARCH, brackets=frozenset({"filter_properties"}))
    ) is not None:
        return refusal
    body, refusal = await _body_object(request)
    if refusal is not None:
        return refusal
    if (refusal := _unknown_body(request, body, _B_SEARCH_MEMBERS)) is not None:
        return refusal
    # Measured against api.notion.com on 2026-10-01 and 2026-10-04: once the credential and the
    # version clear, a `query` that is present must be a string and a `filter` that is present an
    # object, `null` included, or the request is 400 `validation_error` quoting the value back.
    # When both are wrong the message names `query`, wherever each sits in the body.
    for member, kind, named in (("query", str, "a string"), ("filter", dict, "an object")):
        if member in body and not isinstance(body[member], kind):
            value = json.dumps(body[member], ensure_ascii=False, separators=(",", ":"))
            return _error(
                request,
                400,
                "validation_error",
                f"body failed validation: body.{member} should be {named} or `undefined`, "
                f"instead was `{value}`.",
            )
    # A filter and a sort real refuses are named before a row is read (measured 2026-09-30).
    if "filter" in body:
        if (refusal := _search_filter_refusal(request, body, _version(request))) is not None:
            return refusal
    if "sort" in body:
        if (refusal := _search_sort_refusal(request, body)) is not None:
            return refusal
    limit, refusal = _body_page_size(body, "list", request)
    if refusal is not None:
        return refusal
    conn = auth.conn(request)
    visible = auth.visible_ids(request, caller)
    query = body.get("query", "")
    want = body.get("filter", {}).get("value")  # 'page' | 'database' | 'data_source' | None
    version = _version(request)
    # A `data_source` filter keeps the database rows. That filter, an unfiltered search and a
    # `query` list each database as a database before 2025-09-03 and as its data source from it,
    # in the shape `GET /v1/data_sources/{id}` answers less its `request_id`. Real answered the
    # filter with the three databases under 2022-06-28 and their three data sources under
    # 2025-09-03 and 2026-03-11 (2026-10-14), `{}` the same beside the pages (2026-10-14 and
    # 2026-10-17), and a `query` naming one database's title with that database's data source
    # under 2025-09-03 (2026-10-17).
    render_data_source = _data_sources_model(version) and want in (None, "data_source")

    def read(window: int) -> list:
        """The object-type filter applied to one over-fetched window of the corpus. Over-fetching
        keeps a filtered page full; the window is bounded, not the corpus. An empty query means
        "everything the integration can see" (real Notion behavior), so list instead of FTS."""
        if query.strip():
            rows = store.search_documents(conn, query, "notion", visible, limit=window, offset=0)
        else:
            rows = store.list_documents(conn, "notion", visible_ids=visible, limit=window, offset=0)
        return [
            r
            for r in rows
            if not (want in ("database", "data_source") and r["subtype"] != "database")
            and not (want == "page" and r["subtype"] == "database")
        ]

    size = _PAGE_MAX if limit is _NEGATIVE_PAGE else limit
    token = body.get("start_cursor")
    # A uuid this route did not mint (its own cursor is the reserved prefix) names an item, and
    # the first window is where any item a page hands back as its cursor sits.
    ids = []
    window = None
    if token and _own_cursor_offset(str(token)) is None and _UUID_RE.fullmatch(str(token)):
        window = read(size * 4)
        ids = [r["id"] for r in window]
    offset, refusal = _cursor(token, ids, "uuid", request, "body")
    if refusal is not None:
        return refusal
    # A uuid cursor already read the first window to find the item it names; reuse it when the
    # page it resumes at still fits inside, so that request runs one store query, not two.
    picked = (
        window if window is not None and offset + size <= len(window) else read(size * 4 + offset)
    )
    if limit is _NEGATIVE_PAGE:
        return _first_cursor_page(picked[0]["id"] if picked else None, "page_or_database")
    shown = picked[offset : offset + size]
    # A database row is narrowed under every Notion-Version, its parent typed `database_id` or
    # `data_source_id`; a page outside a database keeps its title whatever is listed.
    results = [
        _data_source_obj(conn, r)
        if r["subtype"] == "database" and render_data_source
        else _database_obj(conn, r, version)
        if r["subtype"] == "database"
        else _filter_properties(_page_obj(conn, r), keep if r["parent_id"] else None)
        for r in shown
    ]
    return _list_obj(
        results, offset, len(shown), len(picked), "page_or_database", encode=_uuid_cursor
    )


# --------------------------------------------------------------------------- users


@router.get("/users", response_model=NotionList, openapi_extra=_params(*_P_PAGINATE))
async def list_users(request: Request):
    caller = auth.resolve_bearer(request)
    if (refusal := _refusal(request, caller)) is not None:
        return refusal
    conn = auth.conn(request)
    if (refusal := _unknown_query(request, _Q_PAGINATE)) is not None:
        return refusal
    users = store.list_users(conn)
    ids = [synth.notion_user_id(u["email"]) for u in users]
    limit, refusal = _query_page_size(request, "users")
    if refusal is not None:
        return refusal
    offset, refusal = _cursor(
        request.query_params.get("start_cursor"), ids, "users", request, "query"
    )
    if refusal is not None:
        return refusal
    if limit is _NEGATIVE_PAGE:
        return _first_cursor_page(ids[0] if ids else None, "user")
    page = users[offset : offset + limit]
    results = [_user_obj(conn, u["email"]) for u in page]
    return _list_obj(results, offset, len(page), len(users), "user")


@router.get("/users/me", response_model=NotionObject, openapi_extra=_params())
async def get_me(request: Request):
    caller = auth.resolve_bearer(request)
    if (refusal := _refusal(request, caller)) is not None:
        return refusal
    conn = auth.conn(request)
    if caller.email:  # a user token → that person
        return _user_obj(conn, caller.email)
    # admin/service token → the integration's bot user
    return {
        "object": "user",
        "id": synth.notion_user_id("bot:backlot"),
        "type": "bot",
        "name": "backlot",
        "avatar_url": None,
        "bot": {
            "owner": {"type": "workspace", "workspace": True},
            "workspace_name": auth.acl(request).org_name,
        },
    }


@router.get("/users/{user_id}", response_model=NotionObject, openapi_extra=_params())
async def get_user(user_id: str, request: Request):
    caller = auth.resolve_bearer(request)
    if (refusal := _refusal(request, caller)) is not None:
        return refusal
    conn = auth.conn(request)
    key = _norm(user_id)
    for u in store.list_users(conn):
        if _norm(synth.notion_user_id(u["email"])) == key:
            return _user_obj(conn, u["email"])
    return _error(request, 404, "object_not_found", f"Could not find user with ID: {user_id}.")


# --------------------------------------------------------------------------- comments


@router.get("/comments", response_model=NotionList, openapi_extra=_params(*_P_COMMENTS))
async def list_comments(request: Request):
    caller = auth.resolve_bearer(request)
    if (refusal := _refusal(request, caller)) is not None:
        return refusal
    conn = auth.conn(request)
    block_id = request.query_params.get("block_id")
    # Once the credential and the version clear, real refuses a comments query whose `block_id`
    # is absent, empty or not a uuid before it looks the id up (measured 2026-10-01 and
    # 2026-10-02), and answers a database's id as restricted -- where a page's or a block's id is
    # the 200 list both sides agree on.
    if block_id is None:
        return _error(
            request,
            400,
            "validation_error",
            "query failed validation: query.block_id should be defined, instead was `undefined`.",
        )
    if not block_id.strip():
        return _error(
            request,
            400,
            "validation_error",
            "query failed validation: query.block_id should be a string, instead was `0`.",
        )
    # What real reads as a uuid here (measured 2026-10-02): the dashed 8-4-4-4-12 form in either
    # case, or 32 lower-case hex digits. It echoes anything else as JSON.
    if not re.fullmatch(
        r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}|[0-9a-f]{32}",
        block_id,
    ):
        return _error(
            request,
            400,
            "validation_error",
            "query failed validation: query.block_id should be a valid uuid, instead was "
            f"`{json.dumps(block_id, ensure_ascii=False)}`.",
        )
    # The rest of the query string is judged before the id is looked up, so the derived-block
    # answer below does not skip a member real refuses.
    if (refusal := _unknown_query(request, _Q_COMMENTS)) is not None:
        return refusal
    limit, refusal = _query_page_size(request, "comments")
    if refusal is not None:
        return refusal
    # One ACL-scoped query (the parent must itself be visible to the caller), not
    # a resolve followed by a get_document refetch of the same row -- see get_page and friends.
    row = store.notion_by_id(conn, _norm(block_id), auth.visible_ids(request, caller))
    if row is None:
        # A block is not a row: it is one of the blocks `blocks/{id}/children` derives for a page,
        # and Backlot keeps no comment on one, so a block of a page the caller can see is the empty
        # list real answered on 2026-10-02.
        visible = auth.visible_ids(request, caller)
        wanted = _norm(block_id)
        for page in store.list_documents(conn, "notion", visible_ids=visible, limit=-1):
            if page["subtype"] != "database" and any(
                b["id"] == wanted for b in synth.notion_blocks(page["id"], page["content"])
            ):
                offset, refusal = _cursor(
                    request.query_params.get("start_cursor"), [], "uuid", request, "query"
                )
                if refusal is not None:
                    return refusal
                return _list_obj([], offset, 0, 0, "comment")
        return _error(
            request, 404, "object_not_found", f"Could not find block with ID: {block_id}."
        )
    if row["subtype"] == "database":
        return _error(
            request,
            403,
            "restricted_resource",
            "This object is managed by Notion and isn’t accessible via MCP",
        )
    parent_id = row["id"]
    comments = store.doc_comments(conn, "notion", parent_id)
    ids = [synth.notion_id(c["id"]) for c in comments]
    offset, refusal = _cursor(
        request.query_params.get("start_cursor"), ids, "uuid", request, "query"
    )
    if refusal is not None:
        return refusal
    if limit is _NEGATIVE_PAGE:
        return _first_cursor_page(ids[0] if ids else None, "comment")
    page = comments[offset : offset + limit]
    results = [
        {
            "object": "comment",
            "id": synth.notion_id(c["id"]),
            "parent": {"type": "page_id", "page_id": parent_id},
            "discussion_id": synth.notion_id(f"disc:{row['id']}"),
            "created_time": synth.rfc3339(c["created_ts"]),
            "last_edited_time": synth.rfc3339(c["created_ts"]),
            "created_by": _user_obj(conn, c["author_email"]),
            "rich_text": synth.notion_rich_text(c["body"]),
        }
        for c in page
    ]
    return _list_obj(results, offset, len(page), len(comments), "comment", encode=_uuid_cursor)


# --------------------------------------------------------------------------- misc


# The operations Notion publishes and no route here serves (see ``backlot.published``); a route
# added here wins over ``unmatched_path`` whatever the baseline says.
_PUBLISHED_NOT_SERVED = published.not_served("notion")
# The three operations Notion's document puts behind `basicAuth` rather than the `bearerAuth`
# every other operation inherits (read 2026-09-28): its OAuth client endpoints.
_CLIENT_AUTH_OPERATIONS = frozenset(
    {("POST", "/v1/oauth/token"), ("POST", "/v1/oauth/introspect"), ("POST", "/v1/oauth/revoke")}
)


def _invalid_client(request: Request) -> JSONResponse:
    """An OAuth client endpoint's refusal, in its own shape rather than Notion's envelope.

    Measured 2026-09-28: `POST /v1/oauth/token`, `/v1/oauth/introspect` and `/v1/oauth/revoke`
    each answered 401 `{"error":"invalid_client","request_id":"<uuid>"}` with the id in
    `x-notion-request-id` too and `WWW-Authenticate: Basic realm="OAuth"`, with no header, with
    `Bearer nope` and with `Basic YTpi`. Backlot registers no OAuth client, so there is no header
    this answers any other way."""
    request_id = _request_id(request)
    return JSONResponse(
        status_code=401,
        content={"error": "invalid_client", "request_id": request_id},
        headers={"x-notion-request-id": request_id, "www-authenticate": 'Basic realm="OAuth"'},
    )


@unmatched_router.api_route(
    "/{rest:path}", methods=["GET", "POST", "PATCH", "PUT", "DELETE", "OPTIONS"]
)
async def unmatched_path(request: Request, rest: str) -> JSONResponse:
    """Notion's answer for a URL no route here serves: the credential's refusal where Notion
    publishes the operation, and otherwise the URL's 400, which it gives before it reads anything
    else.

    Measured 2026-09-22 with no credential at all: `/v1/nonexistent_thing/xyz` answers 400
    `invalid_request_url` with `Invalid request URL.`, and so do the same path carrying a token
    that cannot resolve, the same path with no `Notion-Version`, a path outside `/v1` entirely, and
    a POST to an unknown `/v1` path. A path that does exist answers the credential's 401 instead,
    so the URL is the first thing that API checks.

    A published operation exists for that check whether or not Backlot serves it. On 2026-09-28
    each of the 53 the baseline lists as ``missing_operation`` (see ``backlot.published``) answered
    the format's 401 with no header and the token's 401 with `Bearer nope`, the two a served route
    answers, apart from the three OAuth client endpoints and their own refusal (see
    :func:`_invalid_client`). A request that clears the credential and the version on one of the
    other fifty gets the URL's 400 here, as Backlot has no operation to run: that is the gap each of
    those operations acknowledges.

    The method is part of the URL that check reads: `GET /v1/search` and `GET /v1/pages`, both POST
    routes, `DELETE`, `PUT`, `PATCH` and `OPTIONS` on `/v1/users/me`, a GET route, and `PUT` on
    `/v1/pages/{id}`, which Notion publishes for `GET` and `PATCH`, are each that same 400 rather
    than a 405 — which is why this route takes the six methods below rather than the ones the
    routes above happen to declare, and why no `method_not_allowed` envelope is needed for this
    vendor. `TRACE` is left off: real refuses it at the front door with Cloudflare's own `405 Not
    Allowed` HTML page, never reaching the API, so leaving it unrouted keeps the framework's 405
    here rather than trading it for a 400. An invented method such as `FOO` is a bare 501 there,
    and over uvicorn it never reaches this app: the server's parser answers it 400 `Invalid HTTP
    request received.` (both measured 2026-09-28). A `HEAD` arrives as the GET it is rewritten to
    (`backlot.main.answer_head_as_the_get_without_its_body`), and one trailing slash is dropped
    before routing, so a slashed spelling of a served path does not land here
    (`backlot.main.serve_a_slashed_notion_path_as_the_path_without_it`).

    Not reproduced here: real's own root, `/`, a 302 to the marketing site, a page Backlot does not
    serve; a path in another case, which real routes as the lower-case one (`GET /V1/USERS/ME`
    and `PATCH /v1/Pages/{id}` were each the format's 401 on 2026-09-28) and this answers the URL's
    400; and a method outside the six that the front door passes on to the API, `PROPFIND` and
    `QUERY` among them, which real answered the URL's 400 on 2026-09-28 and this answers the
    framework's 405. `/notion/` gets the URL's 400, and `/notion` is Starlette's 307 to it — the one
    path under `/notion` where its slash redirect fires, since this route matches every other.
    """
    path = request.url.path.removeprefix("/notion")
    if published.includes(_PUBLISHED_NOT_SERVED, request.method, path):
        if (request.method, path) in _CLIENT_AUTH_OPERATIONS:
            return _invalid_client(request)
        if (refusal := _refusal(request, auth.resolve_bearer(request))) is not None:
            return refusal
    return _error(request, 400, "invalid_request_url", "Invalid request URL.")
