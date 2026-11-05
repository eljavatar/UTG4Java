#!/usr/bin/env bash


# repos=(Math Gson JacksonCore JacksonDatabind JacksonXml Chart Time Jsoup Mockito)
repos=(Closure Cli Codec Collections Compress Csv JxPath Lang Math Gson JacksonCore JacksonDatabind JacksonXml Chart Time Jsoup Mockito)
# repos=(Csv Cli Lang Chart Gson)
#repos=(Gson)

# -o /000_test_model/generated_tests/results
# -r Merges_generated_tests_all_fixed_prod7
# -d /000_test_model/generated_tests/Merges_generated_tests_all_fixed_prod
# -t /tmp/checkouts

# Inicializamos una variable para el indicador -A
include_A=false

# Check arguments
while getopts ":o:r:d:a:t:A" opt; do
    case $opt in
        o) output_results_dir="$OPTARG"
            ;;
        r) folder_results="$OPTARG"
            ;;
        d) data_tests_dir="$OPTARG"
            ;;
        a) max_attempts="$OPTARG"
            ;;
        t) temp_checkouts_dir="$OPTARG"
            ;;
        A) include_A=true  # Si se pasa -A, activamos la bandera
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
# -o /000_test_model/generated_tests/results 
# -r Merges_generated_tests_all_fixed_prod7 
# -d /000_test_model/generated_tests/Merges_generated_tests_all_fixed_prod/Gson 
# -t /tmp/checkouts
start_time=$(date +%s)

for repo in ${repos[@]}; do
    echo ""
    echo "Executing generated_tests for project: ${repo}..."
    echo ""
    data_tests_dir_project="$data_tests_dir/${repo}"
    echo "data_tests_dir_project: ${data_tests_dir_project}"
    echo ""
    # cmd="perl execute_generated_tests_multiple.pl -s \"$scaffoldings_dir\" -d \"$d4j_data\" -o \"$output_dataset_d4j\""
    # cmd+=" -A"
    # eval $cmd
    # perl execute_generated_tests_multiple.pl -p "$repo" -o "$output_results_dir" -r "$folder_results" -d "$data_tests_dir_project" -t "$temp_checkouts_dir" -V
    if $include_A; then
        perl execute_generated_tests_multiple.pl -p "$repo" -o "$output_results_dir" -r "$folder_results" -d "$data_tests_dir_project" -a "$max_attempts" -t "$temp_checkouts_dir" -A 
    else
        perl execute_generated_tests_multiple.pl -p "$repo" -o "$output_results_dir" -r "$folder_results" -d "$data_tests_dir_project" -a "$max_attempts" -t "$temp_checkouts_dir" 
    fi
    echo "Finish tests execution for project: ${repo}..."
    echo ""
    echo ""
done

end_time=$(date +%s)
elapsed_seconds=$((end_time - start_time))
elapsed_time=$(printf '%02d:%02d:%02d\n' $((elapsed_seconds / 3600)) $(( (elapsed_seconds % 3600) / 60 )) $((elapsed_seconds % 60)))
echo "Total execution time: ${elapsed_time}"
echo ""
