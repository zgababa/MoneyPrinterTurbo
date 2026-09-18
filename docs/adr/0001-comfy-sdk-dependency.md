# Use the official comfy-sdk client for the ComfyUI Cloud provider

Every other paid material provider in this repo (`volcengine_seedance`,
`muapi`, `metaso_minimax`, `ofox`) is hand-rolled against `requests`, with no
vendor SDK dependency. For the ComfyUI Cloud provider we're breaking that
pattern and depending on `comfy-sdk` (`comfy_low.transport.ComfyLow`)
instead of hand-rolling the Jobs/Assets v2 API calls.

We picked this after `openmontage-mcp` (a sibling project integrating the
same ComfyUI Cloud API) documented hitting two real bugs from a hand-rolled
`requests` client — a double-wrapped workflow payload and a wrong assumed
shape for `job.error` — before switching to `comfy-sdk`. Its pydantic models
are generated from the same OpenAPI spec, so a future spec change is more
likely to surface as a typed validation error here than as a silent shape
mismatch. Given that precedent is directly on point, the risk of repeating
those bugs outweighs the value of staying consistent with the rest of the
repo's no-SDK style.
