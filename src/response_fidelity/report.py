"""Portable Markdown reports; no plotting or browser dependency."""

from pathlib import Path


def _text(value):
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def _number(value):
    return "undefined" if value is None else f"{value:.6g}"


def _percent(value):
    return "undefined" if value is None else f"{100 * value:.3f}%"


def markdown(result):
    """Render reports without conflating implementation checks and science claims."""
    lines = ["# Response Fidelity Lab report", "", f"Report type: {_text(result['kind'])}.", ""]
    if result["kind"] == "evaluation":
        lines += [
            f"Evaluation rows: {result['n_samples']}; clusters: {result['n_clusters']}; targets: {result['n_targets']}.",
            "",
            f"Aggregation: {result['weighting']}.",
            "",
            "| Horizon | Nominal MSE | Response MSE | Wrong direction rate | Decision regret |",
            "|---:|---:|---:|---:|---:|",
        ]
        a = result["aggregate"]
        for i, h in enumerate(result["horizons"]):
            lines.append(
                f"| {h:g} | {_number(a['nominal_by_horizon'][i])} | {_number(a['response_by_horizon'][i])} | {_percent(a['wrong_direction_rate_by_horizon'][i])} | {_number(a['decision_regret_by_horizon'][i])} |"
            )
        lines += [
            "",
            f"Overall nominal MSE: {_number(a['nominal_mse'])}; response MSE: {_number(a['response_mse'])}.",
        ]
    elif result["kind"] == "comparison":
        lines += [
            f"Baseline: **{_text(result['baseline'])}**. Positive reductions mean lower error.",
            "",
            "| Candidate | Nominal reduction | Response reduction | Median cluster response reduction | Improved clusters |",
            "|---|---:|---:|---:|---:|",
        ]
        for c in result["comparisons"]:
            r = c["response_mse"]
            lines.append(
                f"| {_text(c['candidate'])} | {_percent(c['nominal_mse']['relative_reduction'])} | {_percent(r['relative_reduction'])} | {_percent(r['median_cluster_relative_reduction'])} | {r['positive_clusters']}/{r['n_clusters']} |"
            )
        lines += [
            "",
            "Relative reductions of aggregate risks and medians of cluster ratios are different summaries.",
            "",
            "## Horizon detail",
            "",
            "| Model | Horizon | Nominal MSE | Response MSE |",
            "|---|---:|---:|---:|",
        ]
        for name, model in result["models"].items():
            for i, h in enumerate(model["horizons"]):
                lines.append(
                    f"| {_text(name)} | {h:g} | {_number(model['aggregate']['nominal_by_horizon'][i])} | {_number(model['aggregate']['response_by_horizon'][i])} |"
                )
        lines += [
            "",
            "## Paired uncertainty",
            "",
            _text(result["bootstrap"]["scope"]),
            "",
            "| Candidate | Response reduction 95% interval |",
            "|---|---|",
        ]
        for c in result["comparisons"]:
            interval = c["response_mse"]["relative_reduction_interval_95"]
            shown = (
                "unavailable"
                if interval is None
                else f"{_percent(interval[0])} to {_percent(interval[1])}"
            )
            lines.append(f"| {_text(c['candidate'])} | {shown} |")
    elif result["kind"] == "geometry":
        lines += [
            f"Mean locally invisible risk: {_number(result['mean_invisible'])}.",
            "",
            f"SVD cutoff: {_number(result['cutoff'])}.",
        ]
        if "mean_total" in result:
            lines += [
                "",
                f"Mean total risk: {_number(result['mean_total'])}; mean visible error: {_number(result['mean_visible'])}.",
                "",
                f"Identity relative error: {_number(result['identity_relative_error'])}.",
            ]
    elif result["kind"] == "noise-oracle":
        lines += ["| Noise standard deviation | Mean oracle risk |", "|---:|---:|"]
        for noise, risk in zip(result["noise"], result["mean_risk_by_noise"], strict=True):
            lines.append(f"| {noise:g} | {_number(risk)} |")
    elif result["kind"] == "head-fit":
        lines += [
            f"Nominal baseline: {_text(result['nominal_candidate'])}; response selection: {_text(result['selected_candidate'])}.",
            "",
            f"Compared {len(result['candidates'])} training-fitted candidates using validation only.",
            "",
            "The report contains no held-out performance claim.",
        ]
    lines += ["", "## Interpretation limits", ""]
    limits = list(result.get("limits", []))
    if result["kind"] == "comparison":
        limits += result["models"][result["baseline"]]["limits"]
    lines += [f"- {_text(limit)}" for limit in limits]
    if "provenance" in result:
        lines += [
            "",
            "## Input provenance",
            "",
            f"Software version: {_text(result['provenance']['software_version'])}.",
            "",
            "| Input | SHA-256 |",
            "|---|---|",
        ]
        for item in result["provenance"]["inputs"]:
            lines.append(f"| {_text(item['file'])} | `{item['sha256']}` |")
    return "\n".join(lines) + "\n"


def write_markdown(path, result):
    text = markdown(result)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text)
