# MLcompiler — Claude Code operating spec

Read VISION.md before every session. It defines the project direction, what we build vs import, and the moat.

Read ROADMAP.md for the runtime arc history and validation rules.

Read logs/session-report.md for the trajectory of work to date.

## Operating rules
- Every commit through full validation cascade
- Parity is non-negotiable
- Two-attempt stop rule on failures
- Write to logs/session-report.md as you go
- No Co-Authored-By trailers
- Stop conditions: validation failures, design decisions affecting architecture, novel bug classes, anything destructive
- Stop and surface decisions before implementing when the change is architectural
