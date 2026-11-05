#!/usr/bin/env bash

# repos=(Closure Cli Codec Collections Compress Csv JxPath Lang Math Gson JacksonCore JacksonDatabind JacksonXml Chart Time Jsoup Mockito)
repos=(Gson Lang Csv Cli Chart)
#repos=(Gson)

# -f /d4j_generated_tests/finetuning_01_codet5p_src_fm_fc_dctx_apply_fix
# -o /d4j_generated_tests/finetuning_01_codet5p_src_fm_fc_dctx_apply_fix
# -s _generated_tests

# Check arguments
while getopts ":f:n:s:" opt; do
    case $opt in
        f) generated_tests_dir="$OPTARG"
            ;;
        n) name_folder_outputs="$OPTARG"
            ;;
        s) suffix_csv_file="$OPTARG"
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
# -f /d4j_generated_tests/finetuning_01_codet5p_src_fm_fc_dctx_apply_fix/Gson_generated_tests.csv
# -o /d4j_generated_tests/finetuning_01_codet5p_src_fm_fc_dctx_apply_fix/Gson
#
start_time=$(date +%s)

for repo in ${repos[@]}; do
    echo ""
    echo "Mergeando generated_tests in template for project: ${repo}..."
    echo ""
    #file_generated_tests="$generated_tests_dir/$repo"
    file_generated_tests_project="$generated_tests_dir/${repo}${suffix_csv_file}.csv"
    output_dir_project="$generated_tests_dir/${name_folder_outputs}/${repo}"
    echo "file_generated_tests_project: ${file_generated_tests_project}"
    echo "output_dir_project: ${output_dir_project}"
    echo ""
    perl merge_tests_in_template_multiple.pl -p "$repo" -f "$file_generated_tests_project" -o "$output_dir_project"
    # perl merge_tests_in_template_multiple.pl -p "$repo" -f "$file_generated_tests_project" -o "$output_dir_project" -V
    echo "Finish generated_tests with template for project: ${repo}..."
    echo ""
    echo ""
done

end_time=$(date +%s)
elapsed_seconds=$((end_time - start_time))
elapsed_time=$(printf '%02d:%02d:%02d\n' $((elapsed_seconds / 3600)) $(( (elapsed_seconds % 3600) / 60 )) $((elapsed_seconds % 60)))
echo "Total execution time: ${elapsed_time}"
echo ""
