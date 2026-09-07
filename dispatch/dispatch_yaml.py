import glob
import itertools
import os

import yaml


def format_value(val):
    if isinstance(val, float):
        if val.is_integer():
            return str(int(val))
        else:
            return str(val).rstrip("0").rstrip(".").replace(".", "")
    # Check if it's a string that looks like a float but has a dot
    if isinstance(val, str) and "." in val:
        try:
            f_val = float(val)
            if f_val.is_integer():
                return str(int(f_val))
            else:
                return str(f_val).rstrip("0").rstrip(".").replace(".", "")
        except ValueError:
            pass
    return str(val)


def process_yaml(yaml_path, dry):
    if not os.path.exists(yaml_path):
        yaml_path = os.path.abspath(
            os.path.join(__file__, "..", "experiments", yaml_path)
        )

    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    ablations = config.get("ablations", {})

    base_config = {k: v for k, v in config.items() if k != "ablations"}
    name_prefix_base = base_config.pop("name_prefix", "experiment")

    keys = list(ablations.keys())
    values_lists = [
        ablations[k] if isinstance(ablations[k], list) else [ablations[k]] for k in keys
    ]

    if not keys:
        combinations = [()]
    else:
        combinations = list(itertools.product(*values_lists))

    commands = []

    for combo in combinations:
        ablation_kwargs = dict(zip(keys, combo))
        curr_config = dict(base_config)
        curr_config.update(ablation_kwargs)

        # Build the name prefix
        suffix_parts = []
        for k, v in ablation_kwargs.items():
            first_letter = k[0]
            val_str = format_value(v)
            suffix_parts.append(f"{first_letter}_{val_str}")

        suffix = "-" + "-".join(suffix_parts) if suffix_parts else ""
        curr_name = f"{name_prefix_base}{suffix}"

        sbatch_time = curr_config.pop("time", "03:00:00")
        sbatch_memo = curr_config.pop("mem", None)

        final_args = {}
        for k, v in curr_config.items():
            if v is not None:
                final_args[k] = v

        # Override name prefix
        final_args["name_prefix"] = curr_name

        cmd_lines = [f"sbatch --time={sbatch_time}"]
        if sbatch_memo is not None:
            cmd_lines.append(f"--mem={sbatch_memo}")
        cmd_lines.append(f"dispatch/dispatch.sh")

        # Format arguments
        for k, v in final_args.items():
            if isinstance(v, bool):
                if v:
                    cmd_lines.append(f"  --{k}")
                elif k == "eval_all":
                    # eval_all defaults to true, so false must be expressed
                    # explicitly rather than omitted from the command.
                    cmd_lines.append("  --no-eval_all")
            elif isinstance(v, list):
                if len(v) == 1 and isinstance(v[0], str) and " " in v[0]:
                    v_str = v[0]
                else:
                    v_str = " ".join(str(x) for x in v)
                cmd_lines.append(f"  --{k} {v_str}")
            else:
                if v == "":
                    cmd_lines.append(f"  --{k}")
                else:
                    cmd_lines.append(f'  --{k} "{v}"')

        command = " \\\n".join(cmd_lines)
        commands.append(command)

    for command in commands:
        print(f"Dispatching{'(dry)' if dry else ''}:\n{command}\n")
        not dry and os.system(command)


def main():
    import sys

    if len(sys.argv) > 1:
        yamls = sys.argv[1:]
    else:
        yamls = glob.glob("experiments_2-8-9/**/*.yaml", recursive=True) + glob.glob(
            "dispatch/experiments_2-8-9/**/*.yaml", recursive=True
        )
        yamls = list(set(yamls))

    dry = "--dry" in yamls

    for y in yamls:
        if y != "--dry":
            process_yaml(y, dry)


if __name__ == "__main__":
    main()
