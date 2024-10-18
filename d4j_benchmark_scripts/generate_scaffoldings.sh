#!/usr/bin/env bash


# Import helper subroutines and variables, and init Defects4J
source /defects4j/framework/test/test.include
init

# Print usage message and exit
usage() {
    local known_pids=$(defects4j pids)
    echo "usage: $0 -p <project id> [-b <bug id> ... | -b <bug id range> ... ]"
    echo "Project ids:"
    for pid in $known_pids; do
        echo "  * $pid"
    done
    exit 1
}

# Check arguments
while getopts ":p:b:" opt; do
    case $opt in
        p) PID="$OPTARG"
            ;;
        b) if [[ "$OPTARG" =~ ^[0-9]*\.\.[0-9]*$ ]]; then
                BUGS="$BUGS $(eval echo {$OPTARG})"
           else
                BUGS="$BUGS $OPTARG"
           fi
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

if [ "$PID" == "" ]; then
    usage
fi

if [ ! -e "$BASE_DIR/framework/core/Project/$PID.pm" ]; then
    usage
fi

init

# Run all bugs, unless otherwise specified
if [ "$BUGS" == "" ]; then
    BUGS="$(get_bug_ids $BASE_DIR/framework/projects/$PID/$BUGS_CSV_ACTIVE)"
fi

# Create log file
path_logs="$TEST_DIR/logs_generate_scaffoldings"
mkdir -p $path_logs
script_name=$(echo $script | sed 's/\.sh$//')
#LOG="$TEST_DIR/${script_name}$(printf '_%s_%s' $PID $$).log"
LOG="$path_logs/${script_name}$(printf '_%s_%s' $PID $$).log"

# Reproduce all bugs (and log all results), regardless of whether errors occur
HALT_ON_ERROR=0

#project_checkout_dir="$TMP_DIR/checkouts"
work_dir="$TMP_DIR/$PID"
mkdir -p $work_dir

# Creamos la carpeta donde copiaremos los scaffoldings generados
scaffoldings_dir="/tmp/scaffoldings/$PID"
mkdir -p $scaffoldings_dir
rm -rf "$scaffoldings_dir/*"

# Clean working directory
rm -rf "$work_dir/*"

for bid in $(echo $BUGS); do
    # Skip all bug ids that do not exist in the active-bugs csv
    if ! grep -q "^$bid," "$BASE_DIR/framework/projects/$PID/$BUGS_CSV_ACTIVE"; then
        warn "Skipping bug ID that is not listed in active-bugs csv: $PID-$bid"
        continue
    fi

    # Use the modified classes as target classes for efficiency
    target_classes="$BASE_DIR/framework/projects/$PID/modified_classes/$bid.src"

    tool="evosuite"
    # Directory for generated test suites
    suite_src="$tool"
    suite_num=1
    suite_dir="$work_dir/$tool/$suite_num"

    # Generate (regression) tests for the fixed version
    vid=${bid}f

    # Run generator and the fix script on the generated test suite
    if ! gen_tests.pl -g "$tool" -p $PID -v $vid -n 1 -o "$TMP_DIR" -t "/tmp/checkouts" -b 300 -c "$target_classes"; then
        die "run $tool (regression) on $PID-$vid"
        # Skip any remaining analyses (cannot be run), even if halt-on-error is false
        continue
    fi

    # Copiamos el archivo de scaffoldings generado en la carpeta destino
    cp "$suite_dir/$PID-$vid-$tool.1.tar.bz2" "$scaffoldings_dir"

    # Extraemos los archivos para limpiar el scaffolding
    tar -xjf "$suite_dir/$PID-$vid-$tool.1.tar.bz2" -C "$suite_dir"

    # Accedemos a los Scaffolding y eliminamos los métodos generados por EvoSuite
    perl clean_scaffolding.pl -c "$target_classes" -s "$suite_dir"

    # Borramos el anterior comprimido
    rm -rf "$suite_dir/$PID-$vid-$tool.1.tar.bz2"

    # Comprimimos de nuevo los archivos
    cd "$suite_dir" # Nos paramos en la ruta donde están los archivos a comprimir
    tar -cjf "$scaffoldings_dir/$PID-$vid-scaffolding.tar.bz2" .

    # Nos paramos de nuevo en el directorio de ejecución
    cd "/defects4j/framework/custom"

    #fix_test_suite.pl -p $PID -d "$suite_dir" || die "fix test suite"

    # Run test suite and determine bug detection
    #test_bug_detection $PID "$suite_dir"

    # Run test suite and determine mutation score
    #test_mutation $PID "$suite_dir"

    # Run test suite and determine code coverage
    #test_coverage $PID "$suite_dir" 0

    rm -rf $work_dir/$tool

done

HALT_ON_ERROR=1

# Print a summary of what went wrong
if [ $ERROR != 0 ]; then
    printf '=%.s' $(seq 1 80) 1>&2
    echo 1>&2
    echo "The following errors occurred:" 1>&2
    cat $LOG 1>&2
fi

# Indicate whether an error occurred
exit $ERROR