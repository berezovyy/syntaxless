let meta = #{
    name: "cron-heads-only",
    description: "Delete cron parser hacks, then one train/export/eval cycle toward 100% heads-only",
    when_to_use: "Syntaxless cron: remove regex overlay then train until overlay-off is 100%",
    phases: [
        #{ title: "Hacks", detail: "delete discourse/correction/force-relabel from the public parser" },
        #{ title: "Train", detail: "one real train.py + export candidate cycle" },
        #{ title: "Score", detail: "heads-only evaluate/gate; 100% or next cycle" },
    ],
};

let report_schema = #{
    "type": "object",
    "required": ["done", "notes"],
    "properties": #{
        "done": #{ "type": "boolean" },
        "notes": #{ "type": "string" },
        "exact_ast": #{ "type": "string" },
        "hundred": #{ "type": "boolean" },
    },
};

phase("Hacks");
let hacks_prompt = "";
hacks_prompt += "Workspace /home/berezovyy/Projects/syntaxless. Read GOAL.md and PROGRESS.md. ";
hacks_prompt += "Public parse must be heads-only. Delete scopeActionableCorrection, unsafeScheduleContext, ";
hacks_prompt += "hasDirectScheduleRequest, unrequestedDeclarativeContext, explicit* role overwrite, ";
hacks_prompt += "and confidence=1 from raw-string hits. Keep lexicon tables and clock math from labels. ";
hacks_prompt += "No overlay-on default. Missing speech-act head is model-error, never regex fallback. ";
hacks_prompt += "Use grep/read_file/edit. Fix tests to the heads-only contract. Do not add decoder regex. ";
hacks_prompt += "Update PROGRESS.md. Return {done, notes}.";

let hacks = agent(hacks_prompt, #{
    label: "delete-hacks",
    capability_mode: "all",
    output_schema: report_schema,
});

if hacks == () || hacks.success != true || hacks.output == () || hacks.output.done != true {
    complete(#{ summary: "hacks step failed or incomplete", hacks: hacks });
}

phase("Train");
let train_prompt = "";
train_prompt += "Workspace /home/berezovyy/Projects/syntaxless. Read GOAL.md and PROGRESS.md. ";
train_prompt += "Run the real training/cron/train.py harness with QAT and speech-act plus actionable heads. ";
train_prompt += "CUDA auto. Improve generator kinds if the last overlay-off misses named a kind gap. ";
train_prompt += "Never copy human-v jsonl wording. Export to packages/cron/training/candidate/weights.gen.ts. ";
train_prompt += "Do not copy to src/model/weights.gen.ts unless evaluate is already 100%. ";
train_prompt += "Update PROGRESS.md. Return {done, notes}.";

let trained = agent(train_prompt, #{
    label: "train-export",
    capability_mode: "all",
    output_schema: report_schema,
});

if trained == () || trained.success != true || trained.output == () || trained.output.done != true {
    complete(#{ summary: "train step failed", hacks: hacks.output, train: trained });
}

phase("Score");
let score_prompt = "";
score_prompt += "Workspace /home/berezovyy/Projects/syntaxless. Read GOAL.md. ";
score_prompt += "Score the candidate heads-only with packages/cron/training/src/evaluate.ts and gate.ts. ";
score_prompt += "Use grep/read_file and run the real commands. ";
score_prompt += "Set hundred=true only if reserved-carrier exactAst, rawDecode, compiledExpression, ";
score_prompt += "and tokenRoles are all 100% on heads-only, silent meaning-change is 0, false-accept is 0. ";
score_prompt += "Put the exactAst fraction in exact_ast. Update PROGRESS.md. Return {done, notes, exact_ast, hundred}.";

let scored = agent(score_prompt, #{
    label: "score-heads",
    capability_mode: "execute",
    output_schema: report_schema,
});

if scored == () || scored.success != true || scored.output == () {
    complete(#{ summary: "score step failed", train: trained.output });
}

complete(#{
    summary: "one heads-only cycle",
    hacks: hacks.output,
    train: trained.output,
    score: scored.output,
    hundred: scored.output.hundred,
});
