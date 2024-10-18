#!/usr/bin/env bash


repos=(Csv Cli Lang Chart Gson)
#repos=(Gson)

# -s /scaffoldings
# -f /d4j_generated_tests/finetuning_01_codet5p_src_fm_fc_dctx_apply_fix
# -o /d4j_generated_tests/finetuning_01_codet5p_src_fm_fc_dctx_apply_fix

# Check arguments
while getopts ":s:f:o:" opt; do
    case $opt in
        s) scaffoldings_dir="$OPTARG"
            ;;
        f) generated_tests_dir="$OPTARG"
            ;;
        o) output_merges_dir="$OPTARG"
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

#generated_tests_dir="/d4j_generated_tests/finetuning_01_codet5p_src_fm_fc_dctx_apply_fix"
#output_merges_dir="/d4j_generated_tests/finetuning_01_codet5p_src_fm_fc_dctx_apply_fix"

# -p Gson 
# -s /scaffoldings/Gson
# -f /d4j_generated_tests/finetuning_01_codet5p_src_fm_fc_dctx_apply_fix/Gson_generated_tests.csv
# -o /d4j_generated_tests/finetuning_01_codet5p_src_fm_fc_dctx_apply_fix/Gson
start_time=$(date +%s)

for repo in ${repos[@]}; do
    echo ""
    echo "Mergeando generated_tests in scaffoldings for project: ${repo}..."
    echo ""
    #file_generated_tests="$generated_tests_dir/$repo"
    scaffoldings_dir_project="$scaffoldings_dir/${repo}"
    file_generated_tests_project="$generated_tests_dir/${repo}_generated_tests.csv"
    output_dir_project="$generated_tests_dir/${repo}"
    echo "scaffoldings_dir_project: ${scaffoldings_dir_project}"
    echo "file_generated_tests_project: ${file_generated_tests_project}"
    echo "output_dir_project: ${output_dir_project}"
    echo ""
    perl merge_tests_in_scaffoldings.pl -p "$repo" -s "$scaffoldings_dir_project" -f "$file_generated_tests_project" -o "$output_dir_project"
    echo "Finish generated_tests in scaffoldings for project: ${repo}..."
    echo ""
    echo ""
done

end_time=$(date +%s)
elapsed_seconds=$((end_time - start_time))
elapsed_time=$(printf '%02d:%02d:%02d\n' $((elapsed_seconds / 3600)) $(( (elapsed_seconds % 3600) / 60 )) $((elapsed_seconds % 60)))
echo "Tiempo total de ejecución: ${elapsed_time}"
echo ""
