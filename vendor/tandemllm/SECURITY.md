# Security policy

## Report a vulnerability

Report it in private on GitHub: open [Report a vulnerability](https://github.com/0xBakeer/TandemLLM/security/advisories/new) on 0xBakeer/TandemLLM. Only you and the maintainer see the report. A public issue or pull request would show it to everyone before a fix exists.

Say which version you ran. `VERSION` in the checkout has it, and so does `GET /v1/models` on a running server. Then name the part of TandemLLM, the request or file that sets it off, and what an attacker gets. A script or a failing test helps most. Please check the latest release first.

## Scope

All code in this repository that a running install exposes is in scope: the engine server on port 8000 and its dashboard, the setup API, the proxy, the Chat service with its sandbox and egress proxy, the installers, and the loaders for weights and images.

An install starts with no admin password, and then every admin route answers anyone who reaches port 8000. [docs/server.md](docs/server.md#the-admin-rule) says so, and the server belongs on a trusted network. An admin route that answers without a password is therefore expected, not a bug. A report that gets more than the route is for, such as running commands or reading files on the box, is in scope. Any way out of a chat's sandbox or past its egress rules is in scope too.

Out of scope: anything that needs a shell on the box, model output quality, numbers that differ in the GPU kernels, third-party weights, and the scripts under `bench/` and `tools/`.

[docs/threat-model.md](docs/threat-model.md) says where untrusted input enters and which parts matter most.
