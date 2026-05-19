# Agent Notes

## Oracle ChatGPT Pro Extended Runs

When a user asks for Oracle with ChatGPT Pro Extended, do not trust generic
browser effort automation by itself. ChatGPT currently has two easy-to-confuse
paths:

- wrong: Thinking model -> Extended thinking
- correct: Pro model -> Extended, shown in the UI as Extended Pro

In this repo this has already caused bad Oracle submissions. The command shape
below can select or report the wrong Extended path if the ChatGPT UI shifts:

```bash
oracle --engine browser --model gpt-5.5-pro --browser-thinking-time extended ...
```

For Pro Extended requests, prefer a manual bundle workflow:

1. Build a stable bundle instead of submitting through browser automation:

   ```bash
   oracle --render --render-plain \
     --prompt "$(cat target/reports/oracle/<prompt>.md)" \
     --file <files...> \
     > target/reports/oracle/<slug>-bundle.md
   ```

2. Open ChatGPT manually.
3. In the model picker, select the Pro model first.
4. Open the Pro row's effort selector and choose Extended.
5. Confirm the composer/model chip reads Extended Pro, not Thinking or plain
   Extended.
6. Paste or upload `target/reports/oracle/<slug>-bundle.md`.
7. Save the answer under `target/reports/oracle/<slug>.md`.

If automation is still required, only use it after the user has manually
selected Extended Pro in the browser. Use current-model preservation and omit
thinking-time changes:

```bash
oracle --engine browser \
  --browser-model-strategy current \
  --browser-tab current \
  --prompt "$(cat target/reports/oracle/<prompt>.md)" \
  --file <files...>
```

Before letting an automated run submit, visually verify that the ChatGPT UI says
Extended Pro. Abort if it says Thinking, Thinking Extended, or only Extended.
Do not claim an Oracle answer came from Pro Extended unless that UI state was
verified before submission.

If an aborted Oracle browser run leaves `oracle status` showing `running` but
`ps` shows no Oracle/Chrome process, treat it as stale local session metadata.
Do not reattach and continue blindly; start a new, correctly configured manual
bundle or verified-current-browser run.
