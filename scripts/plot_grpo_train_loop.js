#!/usr/bin/env node
/*
Create writeup-ready artifacts for the Ch7 on-policy GRPO train-loop run.

This helper intentionally uses only Node's standard library so the archived
figures can be regenerated in a lightweight local environment.
*/

const fs = require("fs");
const path = require("path");

const DEFAULT_RUN_DIR =
  "artifacts/experiments/ch7/grpo_train_loop/runs/grpo_on_policy_lr1e-5";
const DEFAULT_OUTPUT_DIR = "artifacts/experiments/ch7/grpo_train_loop";

function parseArgs(argv) {
  const args = {
    runDir: DEFAULT_RUN_DIR,
    outputDir: DEFAULT_OUTPUT_DIR,
  };

  for (let i = 2; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === "--run-dir") {
      args.runDir = argv[++i];
    } else if (arg === "--output-dir") {
      args.outputDir = argv[++i];
    } else {
      throw new Error(`Unknown argument: ${arg}`);
    }
  }

  return args;
}

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, "utf8"));
}

function readJsonl(filePath) {
  return fs
    .readFileSync(filePath, "utf8")
    .split(/\r?\n/)
    .filter(Boolean)
    .map((line) => JSON.parse(line));
}

function fmtPercent(value) {
  return `${(100 * value).toFixed(2)}%`;
}

function csvEscape(value) {
  const text = String(value);
  if (/[",\n\r]/.test(text)) {
    return `"${text.replace(/"/g, '""')}"`;
  }
  return text;
}

function writeCsv(filePath, header, rows) {
  const lines = [
    header.join(","),
    ...rows.map((row) => row.map(csvEscape).join(",")),
  ];
  fs.writeFileSync(filePath, `${lines.join("\n")}\n`, "utf8");
}

function makeLinearScale(domainMin, domainMax, rangeMin, rangeMax) {
  if (domainMax === domainMin) {
    return () => (rangeMin + rangeMax) / 2;
  }
  return (value) =>
    rangeMin +
    ((value - domainMin) / (domainMax - domainMin)) * (rangeMax - rangeMin);
}

function svgPlot({ title, yLabel, series, yMax, outputPath }) {
  const width = 820;
  const height = 430;
  const margin = { top: 54, right: 36, bottom: 58, left: 72 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const xMax = Math.max(...series.map((point) => point.step));
  const x = makeLinearScale(0, xMax, margin.left, margin.left + plotWidth);
  const y = makeLinearScale(0, yMax, margin.top + plotHeight, margin.top);
  const grid = [];

  for (let i = 0; i <= 5; i += 1) {
    const value = i / 5;
    const yy = y(value);
    grid.push(
      `<line x1="${margin.left}" y1="${yy.toFixed(2)}" x2="${(
        margin.left + plotWidth
      ).toFixed(2)}" y2="${yy.toFixed(
        2
      )}" stroke="#d8dee9" stroke-width="1" />`
    );
    grid.push(
      `<text x="${margin.left - 12}" y="${(yy + 4).toFixed(
        2
      )}" text-anchor="end" font-size="12" fill="#4b5563">${Math.round(
        value * 100
      )}%</text>`
    );
  }

  for (let i = 0; i <= 4; i += 1) {
    const value = Math.round((xMax * i) / 4);
    const xx = x(value);
    grid.push(
      `<line x1="${xx.toFixed(2)}" y1="${margin.top}" x2="${xx.toFixed(
        2
      )}" y2="${(margin.top + plotHeight).toFixed(
        2
      )}" stroke="#edf2f7" stroke-width="1" />`
    );
    grid.push(
      `<text x="${xx.toFixed(2)}" y="${height - 22}" text-anchor="middle" font-size="12" fill="#4b5563">${value}</text>`
    );
  }

  const points = series
    .map((point) => `${x(point.step).toFixed(2)},${y(point.value).toFixed(2)}`)
    .join(" ");
  const circles = series
    .map(
      (point) =>
        `<circle cx="${x(point.step).toFixed(2)}" cy="${y(point.value).toFixed(
          2
        )}" r="3" fill="#2563eb"><title>step ${
          point.step
        }: ${fmtPercent(point.value)}</title></circle>`
    )
    .join("\n");

  const svg = `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" role="img" aria-label="${title}">
  <rect width="100%" height="100%" fill="#ffffff" />
  <text x="${margin.left}" y="30" font-size="20" font-weight="700" fill="#111827">${title}</text>
  ${grid.join("\n  ")}
  <line x1="${margin.left}" y1="${margin.top}" x2="${margin.left}" y2="${margin.top + plotHeight}" stroke="#111827" stroke-width="1.4" />
  <line x1="${margin.left}" y1="${margin.top + plotHeight}" x2="${margin.left + plotWidth}" y2="${margin.top + plotHeight}" stroke="#111827" stroke-width="1.4" />
  <polyline fill="none" stroke="#2563eb" stroke-width="3" points="${points}" />
  ${circles}
  <text x="${margin.left + plotWidth / 2}" y="${height - 6}" text-anchor="middle" font-size="13" fill="#111827">GRPO step</text>
  <text x="18" y="${margin.top + plotHeight / 2}" text-anchor="middle" font-size="13" fill="#111827" transform="rotate(-90 18 ${margin.top + plotHeight / 2})">${yLabel}</text>
</svg>
`;
  fs.writeFileSync(outputPath, svg, "utf8");
}

function firstCorrectRollout(samplePath) {
  if (!fs.existsSync(samplePath)) {
    return null;
  }
  const rows = readJsonl(samplePath);
  return rows.find((row) => row.reward === 1) || rows[0] || null;
}

function truncateResponse(response, maxChars = 700) {
  if (response.length <= maxChars) {
    return response;
  }
  return `${response.slice(0, maxChars).trim()}...`;
}

function main() {
  const args = parseArgs(process.argv);
  fs.mkdirSync(args.outputDir, { recursive: true });

  const metrics = readJsonl(path.join(args.runDir, "metrics.jsonl"));
  const config = readJson(path.join(args.runDir, "config.json"));
  const runSummary = readJson(path.join(args.runDir, "run_summary.json"));
  const evalRows = metrics
    .filter((row) => row.type === "eval")
    .sort((a, b) => a.grpo_step - b.grpo_step);
  const trainRows = metrics
    .filter((row) => row.type === "train")
    .sort((a, b) => a.grpo_step - b.grpo_step);
  const rolloutRows = metrics
    .filter((row) => row.type === "rollout")
    .sort((a, b) => a.grpo_step - b.grpo_step);

  writeCsv(
    path.join(args.outputDir, "grpo_train_loop_eval_points.csv"),
    [
      "grpo_step",
      "answer_accuracy",
      "format_accuracy",
      "reward",
      "num_examples",
      "eval_seconds",
    ],
    evalRows.map((row) => [
      row.grpo_step,
      row.answer_accuracy,
      row.format_accuracy,
      row.reward,
      row.num_examples,
      row.eval_seconds,
    ])
  );

  const selectedSteps = [1, 50, 100, 150, 200];
  const rolloutExamples = selectedSteps
    .map((step) => {
      const samplePath = path.join(
        args.runDir,
        `grpo_step_${String(step).padStart(6, "0")}`,
        "sample_rollouts.jsonl"
      );
      const rollout = firstCorrectRollout(samplePath);
      if (!rollout) {
        return null;
      }
      return { step, ...rollout };
    })
    .filter(Boolean);

  writeCsv(
    path.join(args.outputDir, "grpo_train_loop_rollout_examples.csv"),
    [
      "grpo_step",
      "question_index",
      "rollout_index",
      "reward",
      "format_reward",
      "answer_reward",
      "response_token_length",
      "ground_truth",
      "question",
      "response",
    ],
    rolloutExamples.map((row) => [
      row.step,
      row.question_index,
      row.rollout_index,
      row.reward,
      row.format_reward,
      row.answer_reward,
      row.response_token_length,
      row.ground_truth,
      row.question,
      row.response,
    ])
  );

  svgPlot({
    title: "On-policy GRPO validation answer reward",
    yLabel: "validation answer reward",
    series: evalRows.map((row) => ({
      step: row.grpo_step,
      value: row.answer_accuracy,
    })),
    yMax: 1,
    outputPath: path.join(args.outputDir, "grpo_train_loop_validation_reward.svg"),
  });

  svgPlot({
    title: "On-policy GRPO validation format accuracy",
    yLabel: "validation format accuracy",
    series: evalRows.map((row) => ({
      step: row.grpo_step,
      value: row.format_accuracy,
    })),
    yMax: 1,
    outputPath: path.join(args.outputDir, "grpo_train_loop_format_accuracy.svg"),
  });

  const bestEval = evalRows.reduce((best, row) =>
    row.answer_accuracy > best.answer_accuracy ? row : best
  );
  const finalEval = evalRows[evalRows.length - 1];
  const finalTrain = trainRows[trainRows.length - 1];
  const firstEval = evalRows[0];
  const firstRollout = rolloutRows[0];
  const finalRollout = rolloutRows[rolloutRows.length - 1];

  const runSummaries = [
    {
      run_name: "grpo_on_policy_lr1e-5",
      archived_run_dir: path.join(
        args.outputDir,
        "runs",
        "grpo_on_policy_lr1e-5"
      ),
      learning_rate: config.learning_rate,
      n_grpo_steps: config.n_grpo_steps,
      rollout_batch_size: config.rollout_batch_size,
      group_size: config.group_size,
      loss_type: config.loss_type,
      use_std_normalization: config.use_std_normalization,
      first_eval: {
        grpo_step: firstEval.grpo_step,
        answer_accuracy: firstEval.answer_accuracy,
        format_accuracy: firstEval.format_accuracy,
      },
      best_eval: {
        grpo_step: bestEval.grpo_step,
        answer_accuracy: bestEval.answer_accuracy,
        format_accuracy: bestEval.format_accuracy,
      },
      final_eval: {
        grpo_step: finalEval.grpo_step,
        answer_accuracy: finalEval.answer_accuracy,
        format_accuracy: finalEval.format_accuracy,
      },
      first_rollout: {
        grpo_step: firstRollout.grpo_step,
        answer_accuracy: firstRollout.answer_accuracy,
        format_accuracy: firstRollout.format_accuracy,
        avg_response_token_length: firstRollout.avg_response_token_length,
      },
      final_rollout: {
        grpo_step: finalRollout.grpo_step,
        answer_accuracy: finalRollout.answer_accuracy,
        format_accuracy: finalRollout.format_accuracy,
        avg_response_token_length: finalRollout.avg_response_token_length,
      },
      final_train: {
        grpo_step: finalTrain.grpo_step,
        token_entropy: finalTrain.token_entropy,
        grad_norm: finalTrain.grad_norm,
        loss: finalTrain.loss,
      },
      run_summary: runSummary,
    },
  ];
  fs.writeFileSync(
    path.join(args.outputDir, "run_summaries.json"),
    `${JSON.stringify(runSummaries, null, 2)}\n`,
    "utf8"
  );

  const archiveMd = `# GRPO Train Loop Run Archive

| run | learning rate | steps | initial answer | best answer | final answer | final format | final train answer | final avg length |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| \`grpo_on_policy_lr1e-5\` | \`${config.learning_rate}\` | ${config.n_grpo_steps} | ${fmtPercent(
    firstEval.answer_accuracy
  )} | ${fmtPercent(bestEval.answer_accuracy)} @ ${bestEval.grpo_step} | ${fmtPercent(
    finalEval.answer_accuracy
  )} | ${fmtPercent(finalEval.format_accuracy)} | ${fmtPercent(
    finalRollout.answer_accuracy
  )} | ${finalRollout.avg_response_token_length.toFixed(1)} |

Raw run files are archived under \`artifacts/experiments/ch7/grpo_train_loop/runs/grpo_on_policy_lr1e-5/\`.
`;
  fs.writeFileSync(
    path.join(args.outputDir, "run_summaries_archive.md"),
    archiveMd,
    "utf8"
  );

  const examplesMd = [
    "# GRPO Train Loop Rollout Examples",
    "",
    ...rolloutExamples.flatMap((row) => [
      `## Step ${row.step}`,
      "",
      `Question: ${row.question}`,
      "",
      `Ground truth: \`${row.ground_truth}\``,
      "",
      `Reward: ${row.reward}, format reward: ${row.format_reward}, answer reward: ${row.answer_reward}, response tokens: ${row.response_token_length}`,
      "",
      "Response:",
      "",
      "```text",
      truncateResponse(row.response),
      "```",
      "",
    ]),
  ].join("\n");
  fs.writeFileSync(
    path.join(args.outputDir, "grpo_train_loop_rollout_examples.md"),
    examplesMd,
    "utf8"
  );

  const summaryMd = `# GRPO Train Loop Summary

| metric | value |
|---|---:|
| initial validation answer reward | ${fmtPercent(firstEval.answer_accuracy)} |
| best validation answer reward | ${fmtPercent(bestEval.answer_accuracy)} at step ${bestEval.grpo_step} |
| final validation answer reward | ${fmtPercent(finalEval.answer_accuracy)} |
| final validation format accuracy | ${fmtPercent(finalEval.format_accuracy)} |
| final rollout answer reward | ${fmtPercent(finalRollout.answer_accuracy)} |
| final rollout format accuracy | ${fmtPercent(finalRollout.format_accuracy)} |
| final average response token length | ${finalRollout.avg_response_token_length.toFixed(1)} |
| final token entropy | ${finalTrain.token_entropy.toFixed(4)} |
| final gradient norm | ${finalTrain.grad_norm.toFixed(4)} |
`;
  fs.writeFileSync(
    path.join(args.outputDir, "grpo_train_loop_summary.md"),
    summaryMd,
    "utf8"
  );
}

main();
