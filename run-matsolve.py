import json
import re
import subprocess
import sys
import os

int_max = 2147483647
args = sys.argv
parsed_data = {}
run_name = args[1]
executable_path = "./" + run_name
bandwidth_test = ""
if len(args) > 2:
    bandwidth_test = "-" + args[2]


def run_test(arguments):
    global parsed_data
    command = [executable_path] + arguments
    result = subprocess.run(command, capture_output=True, text=True)
    print(result.stdout)
    stdout = result.stdout.split("\n")
    if result.stderr:
        print(result.stderr)
        return True

    stencils = {
        0: "stencilstar",
        1: "stencilbox",
        2: "stencilstarfill1",
        3: "stencildiamond",
    }
    stencil, stencil_width, dof, m, n, p = map(int, arguments)
    problem = stencils[stencil] + ",width=%d" % stencil_width
    dof_str = "dof=%d" % dof
    lower_pattern = re.compile(r"\s+Lower:(\d+),([\d\.]+),([\d\.]+),([\d\.]+)")
    err_pattern = re.compile(r".*out of memory.*")
    if problem not in parsed_data:
        parsed_data[problem] = {}
    if dof_str not in parsed_data[problem]:
        parsed_data[problem][dof_str] = []
    for line in stdout:
        err_match = err_pattern.match(line)
        if err_match:
            return True
        lower_match = lower_pattern.match(line)
        if lower_match:
            lower_data = tuple(map(float, lower_match.groups()))
            parsed_data[problem][dof_str].append(
                {"mesh_size": (m, n, p), "lower": lower_data}
            )
    return False


def generate_test(stencil_type, width, dof):
    k = 32
    while k**3 < int_max:
        if run_test(list(map(str, [stencil_type, width, dof, k, k, k]))):
            break
        print("mesh size:", k)
        k += 16


if bandwidth_test == "":
    generate_test(0, 0, 1)
    generate_test(0, 1, 1)
    generate_test(1, 0, 1)
    generate_test(2, 0, 1)
    generate_test(3, 1, 1)
    generate_test(0, 0, 4)
    generate_test(0, 1, 4)
    generate_test(1, 0, 4)
    generate_test(2, 0, 4)
    generate_test(3, 1, 4)
else:
    if os.environ.get("CUDAARCHS") == 80:
        mesh_size = [
            [560, 416, 320, 256, 224, 192, 176, 160],  # stencilstar,width=0
            [544, 336, 256, 208, 176, 160, 144, 128],  # stencilstar,width=1
            [416, 256, 192, 160, 144, 128, 112, 104],  # stencilbox,width=0
            [544, 336, 256, 208, 176, 160, 144, 128],  # stencilstarfill1,width=0
            [416, 256, 192, 160, 144, 128, 112, 104],  # stencildiamond,width=1
        ]  # dof=1~8
    else:
        mesh_size = [
            [512, 336, 256, 208, 176, 160, 144, 128],  # stencilstar,width=0
            [432, 272, 208, 160, 144, 128, 112, 96],  # stencilstar,width=1
            [336, 208, 160, 128, 112, 96, 80, 80],  # stencilbox,width=0
            [432, 272, 208, 160, 144, 128, 112, 96],  # stencilstarfill1,width=0
            [336, 208, 160, 128, 112, 96, 80, 80],  # stencildiamond,width=1
        ]  # dof=1~8
    dof = list(range(1, 9))
    problems = [(0, 0), (0, 1), (1, 0), (2, 0), (3, 1)]
    for i in range(len(dof)):
        for j in range(5):
            run_test(
                list(
                    map(
                        str,
                        [
                            *problems[j],
                            dof[i],
                            mesh_size[j][i],
                            mesh_size[j][i],
                            mesh_size[j][i],
                        ],
                    )
                )
            )

with open(
    "./results/" + run_name + bandwidth_test + ".json", "w", encoding="utf-8"
) as fout:
    json.dump(parsed_data, fout, indent=4)
