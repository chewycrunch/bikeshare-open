#!/usr/bin/env python3
"""PreToolUse hook on Bash: block commands that would write into data/raw/.

The deny rules in settings.json stop the Edit and Write tools, and Claude
Code also applies them to the destination of a plain `cp`. They do not see
inside an interpreter: `python -c "shutil.copy(..., 'data/raw/...')"` went
straight through (docs/SEEDED_FAULTS.md, fault 4), and the PostToolUse hook
never sees Bash at all. Exit 2 blocks the call and hands stderr to the agent.

This is a heuristic, not a sandbox: a script that writes the file without
naming it on the command line gets past it. The backstop is the SHA-256 pin
in src/config.py, checked by `make data` and tests/test_contract.py.
"""
import json
import re
import shlex
import sys

RAW = "data/raw"
COPIERS = {"cp", "rsync", "install", "ln"}  # only the destination matters
WRITERS = {"mv", "rm", "unlink", "shred", "truncate", "touch", "tee"}  # any data/raw argument
IN_PLACE = {"sed", "gsed", "perl"}  # writers only with -i
INTERPRETERS = {"python", "python3", "perl", "ruby", "node", "Rscript"}
WRITE_CALL = re.compile(
    r"to_csv|to_parquet|shutil\.|os\.(rename|replace|remove|unlink)|\.write|open\([^)]*['\"][wax]"
)
SEPARATORS = {";", "&&", "||", "|", "&", "(", ")"}
REDIRECTS = {">", ">>", ">|", "&>", "&>>"}
PREFIXES = {"sudo", "env", "command", "nohup", "time"}
HEREDOC = re.compile(r"<<-?[ \t]*(['\"]?)(\w+)\1([^\n]*)\n(.*?)\n[ \t]*\2[ \t]*(?=\n|$)", re.S)
ROUGH_TOKENS = re.compile(r"&&|\|\||>>|[;|&()<>]|[^\s;|&()<>]+")


def cut_heredocs(cmd: str) -> tuple[str, list[str]]:
    """Lift heredoc bodies out of the command. A body is data, not shell:
    prose in it (an apostrophe, a mention of data/raw) must not be parsed as
    commands. Bodies are kept so code fed to an interpreter through one
    (`python - <<EOF`) is still checked."""
    bodies: list[str] = []

    def cut(m: re.Match) -> str:
        bodies.append(m.group(4))
        return f" __heredoc{len(bodies) - 1}__ {m.group(3)}"

    return HEREDOC.sub(cut, cmd), bodies


def segments(cmd: str):
    cmd = cmd.replace("\\\n", " ").replace("\n", " ; ")
    try:
        lex = shlex.shlex(cmd, posix=True, punctuation_chars=True)
        lex.whitespace_split = True
        tokens = list(lex)
    except ValueError:  # unbalanced quote: fall back to a rough split
        tokens = ROUGH_TOKENS.findall(cmd)
    seg: list[str] = []
    for tok in tokens:
        if tok in SEPARATORS:
            if seg:
                yield seg
            seg = []
        else:
            seg.append(tok)
    if seg:
        yield seg


def why_blocked(cmd: str) -> str | None:
    if RAW not in cmd:
        return None
    shell, bodies = cut_heredocs(cmd)
    for seg in segments(shell):
        for tok, nxt in zip(seg, seg[1:]):
            if tok in REDIRECTS and RAW in nxt:
                return f"redirects output into {nxt}"
        words = list(seg)
        while words and (re.match(r"^\w+=", words[0]) or words[0] in PREFIXES):
            words.pop(0)
        if not words:
            continue
        verb, args = words[0].rsplit("/", 1)[-1], words[1:]
        paths = [a for a in args if not a.startswith("-")]
        if verb in COPIERS and paths and (RAW in paths[-1] or any(
                RAW in b for a, b in zip(args, args[1:]) if a == "-t")):
            return f"`{verb}` targets data/raw"
        if verb in WRITERS and any(RAW in a for a in args):
            return f"`{verb}` modifies a file under data/raw"
        if verb == "dd" and any(a.startswith("of=") and RAW in a for a in args):
            return "`dd` writes into data/raw"
        if verb in IN_PLACE and any(a.startswith(("-i", "--in-place")) for a in args) \
                and any(RAW in a for a in args):
            return f"`{verb} -i` edits a file under data/raw in place"
        if verb in INTERPRETERS:
            fed = [bodies[int(i)] for a in args for i in re.findall(r"__heredoc(\d+)__", a)]
            code = " ".join(args + fed)
            if RAW in code and WRITE_CALL.search(code):
                return f"`{verb}` code mentions data/raw and writes a file"
    return None


if __name__ == "__main__":
    command = json.load(sys.stdin).get("tool_input", {}).get("command", "")
    reason = why_blocked(command)
    if reason:
        print(
            f"Blocked: this command {reason}. data/raw/ is immutable "
            "(CLAUDE.md); make the change downstream, in code. If the raw "
            "data really must change, a human does it and updates RAW_SHA256 "
            f"in src/config.py.\nCommand: {command}",
            file=sys.stderr,
        )
        sys.exit(2)
    sys.exit(0)
