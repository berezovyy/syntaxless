let meta = #{
    name: "cron-discourse",
    description: "Implement one cron discourse phase from GOAL.md and verify it on the real parse path",
    when_to_use: "After GOAL.md: run the next tmp/cron-discourse phase without adding decoder regex",
    phases: [
        #{ title: "Implement", detail: "read GOAL.md and the phase file; ship the behavior" },
        #{ title: "Verify", detail: "run the phase tests; fail closed on regex slop" },
    ],
};

let report_schema = #{
    "type": "object",
    "required": ["phase", "done", "tests", "notes"],
    "properties": #{
        "phase": #{ "type": "string" },
        "done": #{ "type": "boolean" },
        "tests": #{ "type": "string" },
        "notes": #{ "type": "string" },
    },
};

let phase_id = if args == () { () } else { args.phase };
if phase_id == () {
    pause("verification", "Pass args.phase as \"01\" .. \"14\" matching tmp/cron-discourse/NN-*.md");
}

let phase_file = "tmp/cron-discourse/" + phase_id + "-";
let goal = "GOAL.md";
let design = "tmp/cron-discourse/design.md";
let overview = "tmp/cron-discourse/00-overview.md";

phase("Implement");

let impl_prompt = "";
impl_prompt += "Workspace /home/berezovyy/Projects/syntaxless. You implement ONE phase of GOAL.md. ";
impl_prompt += "Read GOAL.md, tmp/cron-discourse/design.md, tmp/cron-discourse/00-overview.md, PROGRESS.md, ";
impl_prompt += "and the phase file whose name starts with tmp/cron-discourse/";
impl_prompt += phase_id;
impl_prompt += "-. Then read the current source listed in that phase. ";
impl_prompt += "Implement exactly that phase. Use read_file, grep, and the real parser/training files before editing. ";
impl_prompt += "Do not add decoder regex or copy human-v jsonl wording. Overlay-on stays public default until phase 10. ";
impl_prompt += "Do not start the next phase. Run the focused tests named in the phase file. Update PROGRESS.md. ";
impl_prompt += "Return {phase, done, tests, notes}.";

let impl = agent(impl_prompt, #{
    label: "implement-" + phase_id,
    capability_mode: "all",
    output_schema: report_schema,
});

if impl == () || impl.success != true || impl.output == () {
    complete(#{ summary: "implementer failed", phase: phase_id, done: false });
}

phase("Verify");

let ver_prompt = "";
ver_prompt += "Workspace /home/berezovyy/Projects/syntaxless. Adversarially verify phase ";
ver_prompt += phase_id;
ver_prompt += " of GOAL.md. Read the phase file under tmp/cron-discourse/, GOAL.md, the git diff, and the tests. ";
ver_prompt += "Use grep and read_file. Set done=true only if every item in What to test is evidenced in source or test output. ";
ver_prompt += "Set done=false if a new decoder regex, bakery/flights pattern, or consumed-eval wording was added, ";
ver_prompt += "or if overlay-off happy path every weekday at 9 is broken, or if overlay was deleted before phase 10. ";
ver_prompt += "Run the focused bun tests if needed. Return {phase, done, tests, notes}.";

let ver = agent(ver_prompt, #{
    label: "verify-" + phase_id,
    capability_mode: "execute",
    output_schema: report_schema,
});

if ver == () || ver.success != true || ver.output == () {
    complete(#{
        summary: "verifier failed",
        phase: phase_id,
        implementer: impl.output,
        done: false,
    });
}

complete(#{
    summary: "phase " + phase_id,
    implementer: impl.output,
    verifier: ver.output,
    done: ver.output.done,
});
