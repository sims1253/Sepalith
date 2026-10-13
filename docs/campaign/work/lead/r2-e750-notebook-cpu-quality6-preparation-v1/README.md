# E750 six-core notebook offline quality suite

This is a fresh full DEV75 × cap192/384/768 run. It uses all six physical CPU cores (`0,2,4,6,8,10`), 300 seconds per case, context 4096, and the same E750 Q8, PRM-03 protocol, tokenizer, prompt text/IDs, deterministic request settings, and cap accounting as the earlier two-thread preparation. The production reference remains five seconds and is not changed.

The earlier two-thread 21-row prefix is retained only as diagnostic evidence and is never loaded or merged. All three new arms start from row one in a fresh namespace. The separately frozen watchdog enforces 28,800 seconds, records the runner identity, allows 150 seconds for an active request and normal cleanup after TERM, then cleans only matching server process groups from arm launch records.

No model copy or full model rehash was repeated. Preflight relies on the previously verified full SHA plus immutable device/inode/size/time/mode identity. Root must review the complete new closure, fill and stage a fresh admission, and launch the watchdog command.
