#!/usr/bin/env bash


repos=(Csv Cli Lang Chart Gson)
#repos=(Gson)

# -o /d4j_generated_tests/finetuning_01_codet5p_src_fm_fc_dctx_no_apply_fix 
# -d /d4j_generated_tests/finetuning_01_codet5p_src_fm_fc_dctx_no_apply_fix 
# -t /tmp/checkouts

# Check arguments
while getopts ":o:d:t:" opt; do
    case $opt in
        o) output_results_dir="$OPTARG"
            ;;
        d) data_tests_dir="$OPTARG"
            ;;
        t) temp_checkouts_dir="$OPTARG"
            ;;
        \?)
            echo "Unknown option: -$OPTARG" >&2
            usage
            ;;
        :)
            echo "No argument provided: -$OPTARG." >&2
            usage
            ;;
  esac
done

# -p Gson 
# -o /d4j_generated_tests/finetuning_01_codet5p_src_fm_fc_dctx_no_apply_fix 
# -d /d4j_generated_tests/finetuning_01_codet5p_src_fm_fc_dctx_no_apply_fix/Gson 
# -t /tmp/checkouts
start_time=$(date +%s)

for repo in ${repos[@]}; do
    echo ""
    echo "Executing generated_tests for project: ${repo}..."
    echo ""
    data_tests_dir_project="$data_tests_dir/${repo}"
    echo "data_tests_dir_project: ${data_tests_dir_project}"
    echo ""
    perl execute_generated_tests.pl -p "$repo" -o "$output_results_dir" -d "$data_tests_dir_project" -t "$temp_checkouts_dir"
    echo "Finish tests execution for project: ${repo}..."
    echo ""
    echo ""
done

end_time=$(date +%s)
elapsed_seconds=$((end_time - start_time))
elapsed_time=$(printf '%02d:%02d:%02d\n' $((elapsed_seconds / 3600)) $(( (elapsed_seconds % 3600) / 60 )) $((elapsed_seconds % 60)))
echo "Tiempo total de ejecución: ${elapsed_time}"
echo ""
