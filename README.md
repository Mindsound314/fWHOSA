# fWHOSA legacy modules (`v0.0-legacy`)

The five modules behind analyses run before the fwhosa 0.1.0 rewrite, as supplied for the
audit on 9 Nov 2026. They are kept here so that those results can be reproduced.

Do not use them for new work. The
[0.1.0 changelog](https://github.com/Mindsound314/fWHOSA/blob/main/CHANGELOG.md) lists the
defects the rewrite fixes, among them bicoherence that collapses on inputs in volts, edge
artifacts every 4.5 s from 5-s outer windows, and a `c0` that never reached the transform.
The maintained package is on the `main` branch.
