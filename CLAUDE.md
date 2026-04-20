# Claude repo rules

- Preserve existing local work.
- Do not overwrite, remove, or revert user changes unless explicitly asked.
- Keep edits limited to the minimum file set required for the requested task.
- Ignore unrelated modified files.
- Use small diffs instead of broad rewrites.
- Do not refactor shared code unless the request requires it.
- If a change would affect in-progress work outside the current task, stop and ask.
- If the repo is being used for multiple parallel features, recommend a separate branch or `git worktree`.
- At the end, list the files changed.
- If validation was not run, say so.

# Preferred working style

- One feature per branch.
- Separate folders for parallel work using `git worktree`.
- Checkpoint commits before risky changes.
