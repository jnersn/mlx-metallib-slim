# Contributing

This tool is maintained for the needs of the app that uses it. Contributions are welcome, but there is no promised roadmap, support window or
response time.

Please open an issue before starting anything beyond a small fix, so the scope can be agreed first. A change to what the reduction computes
(the filter, the replayed commands, the source edits) changes the bytes of every reduced library built with it: say so in the pull request and
in `CHANGELOG.md`.

Run the tests with `python3 -m unittest discover -s tests` before sending a change. They need no GPU, network or Xcode.

By contributing you agree that your contribution is licensed under the MIT License of this repository (see `LICENSE`).
