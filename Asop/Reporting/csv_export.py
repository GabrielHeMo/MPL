"""CSV export functions extracted from optimizer_runner.py without changing their behavior."""

def write_history_csv_mono(config, results_path, history):
    import os
    import csv

    csv_path = os.path.splitext(results_path)[0] + "_evaluations.csv"

    input_tags = [
        v.get("tag") or v.get("alias") or f"x{i + 1}"
        for i, v in enumerate(config.get("inputs", []))
    ]

    output_tags = [
        v.get("tag") or v.get("alias") or f"y{i + 1}"
        for i, v in enumerate(config.get("outputs", []))
    ]

    objectives_config = config.get("objectives", []) or []

    if len(objectives_config) > 0:
        obj = objectives_config[0]
        objective_name = (
            obj.get("tag")
            or obj.get("alias")
            or obj.get("name")
            or obj.get("objective_name")
            or "obj1"
        )
    else:
        objective_name = "obj1"

    extra_keys = [
        "algorithm",
        "acquisition_function_raw",
        "acquisition_function",
        "turbo_iteration",
        "candidate_source",
        "trust_region_length",
        "acq_value",
        "tr_lb",
        "tr_ub",
        "feasible",
        "aspen_converged",
        "run_status",
    ]

    header = (
        ["evaluation"]
        + [f"variables.{tag}" for tag in input_tags]
        + [f"outputs.{tag}" for tag in output_tags]
        + [f"objectives.{objective_name}"]
        + extra_keys
        + ["message"]
    )

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for row in history:
            x_dict = row.get("x", {}) or {}
            outputs_dict = row.get("outputs", {}) or {}

            variable_values = [
                x_dict.get(tag)
                for tag in input_tags
            ]

            output_values = [
                outputs_dict.get(tag)
                for tag in output_tags
            ]

            extra_values = [
                row.get(key)
                for key in extra_keys
            ]

            writer.writerow(
                [row.get("evaluation")]
                + variable_values
                + output_values
                + [row.get("objective")]
                + extra_values
                + [row.get("message")]
            )

    return csv_path


def write_history_csv_multi(config, results_path, history):
    import os
    import csv

    csv_path = os.path.splitext(results_path)[0] + "_evaluations.csv"

    input_tags = [
        v.get("tag") or v.get("alias") or f"x{i + 1}"
        for i, v in enumerate(config.get("inputs", []))
    ]

    output_tags = [
        v.get("tag") or v.get("alias") or f"y{i + 1}"
        for i, v in enumerate(config.get("outputs", []))
    ]

    objectives_config = config.get("objectives", []) or []

    objective_names = []

    for i, obj in enumerate(objectives_config):
        name = (
            obj.get("tag")
            or obj.get("alias")
            or obj.get("name")
            or obj.get("objective_name")
            or f"obj{i + 1}"
        )
        objective_names.append(str(name))

    if len(objective_names) == 0:
        objective_names = ["obj1"]

    seen = {}

    for i, name in enumerate(objective_names):
        if name not in seen:
            seen[name] = 1
        else:
            seen[name] += 1
            objective_names[i] = f"{name}_{seen[name]}"

    header = (
                ["evaluation"]
                + [f"variables.{tag}" for tag in input_tags]
                + [f"outputs.{tag}" for tag in output_tags]
                + [f"objectives.{name}" for name in objective_names]
                + ["feasible", "message"]
                )

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for row in history:
            x_dict = row.get("x", {}) or {}
            outputs_dict = row.get("outputs", {}) or {}

            variable_values = [
                x_dict.get(tag)
                for tag in input_tags
            ]

            output_values = [
                outputs_dict.get(tag)
                for tag in output_tags
            ]

            objective_values = row.get("objective_values", [])

            if objective_values is None:
                objective_values = []

            objective_values = list(objective_values)

            while len(objective_values) < len(objective_names):
                objective_values.append(None)

            objective_values = objective_values[:len(objective_names)]

            writer.writerow(
                [row.get("evaluation")]
                + variable_values
                + output_values
                + objective_values
                + [
                    row.get("feasible"),
                    row.get("message"),
                ]
            )

    return csv_path

# Auviliary functions
