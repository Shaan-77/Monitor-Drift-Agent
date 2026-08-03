# ZeroUI ER-053 Only UAT — Resolver Files

Use this folder after the blocker is proven.

Behavior:
- `zeroui_uat/insecure_eval_demo.py` replaces `eval()` with `json.loads()`.
- The reusable ZeroUI action uses `http.client` and does not use `urllib.request.urlopen`.
- Bandit should return no findings for the changed files.
- ZeroUI should map `ci.security_scan.failed` to `ER-053`.
- Expected decision: `pass`.
- Expected Active Blocker: ER-053 cleared.

Do not click manual Resolve in TAP for this proof.
