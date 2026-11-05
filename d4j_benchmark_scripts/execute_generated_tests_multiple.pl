#!/usr/bin/env perl

use warnings;
use strict;

use FindBin;
use File::Basename;
use Cwd qw(abs_path);
use Getopt::Std;
use Pod::Usage;
use Text::CSV;
use File::Find;
use File::Copy;
use File::Path qw(make_path);
#make_path($destino) or die "No se pudo crear el directorio de destino '$destino': $!\n"; # Crea directorios recursivamente
use Data::Dumper;
# Install module Text::CSV
# cpan Text::CSV
use IPC::Open3; # Para ejecutar un proceso y capturar su salida
use Symbol 'gensym';
use JSON::PP;  # Módulo para trabajar con JSON

use lib abs_path("$FindBin::Bin/../core");
use Coverage;
use Constants;
use Project;
use Utils;
use Log;

use Time::HiRes qw(gettimeofday tv_interval);


#
# Process arguments and issue usage message if necessary.
#
my %cmd_opts;
getopts('p:d:o:r:v:s:m:t:f:a:AVSD', \%cmd_opts) or pod2usage(1);

pod2usage(1) unless defined $cmd_opts{p} and 
                    defined $cmd_opts{d} and 
                    defined $cmd_opts{o};


my $SUITE_DIR = abs_path($cmd_opts{d}); # /generated-tests/Gson
my $OUT_DIR = abs_path($cmd_opts{o});
my $PID = $cmd_opts{p};
my $FOLDER_RESULTS = $cmd_opts{r} if defined $cmd_opts{r};
my $VID = $cmd_opts{v} if defined $cmd_opts{v};
my $TEST_SRC = $cmd_opts{s} if defined $cmd_opts{s};
my $MUT_OPS_FILE = $cmd_opts{m} if defined $cmd_opts{m}; # Lista de operaciones mutantes. Vacío si se quiere ejecutar todas
my $INCL = $cmd_opts{f} // "*.java";
my $MAX_ATTEMPTS = $cmd_opts{a} if defined $cmd_opts{a};
# my $RM_ASSERTS = defined $cmd_opts{E} ? 1 : 0;
my $ANALYZE_ALL_TEST_SUITES = defined $cmd_opts{A} ? 1 : 0;
my $VALIDATE_TEMPLATE = defined $cmd_opts{V} ? 1 : 0;
my $USE_SCAFFOLDING = defined $cmd_opts{S} ? 1 : 0;
# my $USE_SCAFFOLDING = defined $cmd_opts{S} ? 1 : 0;
# Enable debugging if flag is set
$DEBUG = 1 if defined $cmd_opts{D};

# Guarda el tiempo de inicio
my $start_time = [gettimeofday];

# Check format of target version id
if (defined $VID) {
    Utils::check_vid($VID);
}

my @list_tests;
opendir(DIR, $SUITE_DIR) or die "Could not open directory: $SUITE_DIR!";
my @entries = readdir(DIR);
closedir(DIR);

# Leemos todos los archivos .tar.bz que contienen los tests, los cuales deben tener la siguiente forma:
# {Project}-{version[f|b]}-generated_test.{method_id or test_id}.{suite_id}.tar.bz
# Por ejemplo: Gson-1f-generated_test.0.1.tar.bz
foreach (@entries) {
    #next unless /^([^-]+)-(\d+[bf])-([^\.]+)(\.(\d+))?\.tar\.bz2$/;
    # Gson-1f-scaffolding.tar.bz2
    #next unless /^([^-]+)-(\d+[bf])-([^\.]+)?\.tar\.bz2$/;
    # Gson-1f-generated_test-1.tar.bz2
    #next unless /^([^-]+)-(\d+[bf])-([^\.]+)-(\d+)?\.tar\.bz2$/;
    # Gson-1f-generated_test.1.tar.bz2
    next unless /^([^-]+)-(\d+[bf])-([^\.]+)\.(\d+)\.(\d+)?\.tar\.bz2$/;
    my $pid = $1;
    my $vid = $2;
    #my $src = $3;
    #my $src = "generated-tests";
    my $src = "generated_tests";
    #my $tid = ($5 // "1");
    #my $tid = "1";
    my $tid = $4; # test_id
    my $sid = $5; # suite_id
    # Check whether target pid matches
    #my $origin_file = "${pid}-${vid}-generated_test.${tid}.tar.bz2";
    #my $origin_file = "${pid}-${vid}-generated_test-${tid}.tar.bz2":

    next if ($PID ne $pid);
    # Check whether a target src is defined
    next if defined($TEST_SRC) and ($TEST_SRC ne $src);
    # Check whether a target version_id is defined
    next if defined($VID) and ($VID ne $vid);

    # $src = "generated_test_$sid";
    $src = "generated_test_${vid}_${tid}_${sid}";

    my $vid_num = $vid;
    $vid_num =~ s/[^0-9]//g; # Reemplaza la parte no nuérica por una cadena vacía
    #$vid_num = int($vid_num);
    #print "\n\n vid: $vid --- vid_num: $vid_num \n\n";

    push(@list_tests, {file_name => $_, pid => $pid, vid=>$vid, vid_num=>$vid_num, src=>$src, tid=>$tid, sid=>$sid});
}

# Ordenamos la lista
#@list = sort { $a->{file_name} cmp $b->{file_name} } @list;
@list_tests = sort {
    # Comparar por 'version' primero
    #$a->{vid} cmp $b->{vid} # Si es String, se compara con cmp, si es número, se compara con <=>
    $a->{vid_num} <=> $b->{vid_num} # Primero ordenamos or version
        || $a->{tid} <=> $b->{tid}  # Luego por número de método o test
        || $a->{sid} <=> $b->{sid}  # Y finalmente por suite_id
} @list_tests;


# Set up project
my $TMP_DIR = Utils::get_tmp_dir($cmd_opts{t}); # This is $WORK_DIR
system("mkdir -p $TMP_DIR");


if (!defined $FOLDER_RESULTS) {
    $FOLDER_RESULTS = "results";
}

my $PATH_LOGS = "$OUT_DIR/$FOLDER_RESULTS";
my $PATH_COVERAGE_FILES = "$PATH_LOGS/coverage_files/${PID}";
my $PATH_MUTATION_FILES = "$PATH_LOGS/mutation_files/${PID}";
my $PATH_CORRECT_TESTS = "$PATH_LOGS/correct_tests/${PID}";
my $PATH_INSTRUMENTS = "/tmp/instruments/${PID}";
my $PATH_CORRECT_TESTS_TMP = "/tmp/correct_tests/${PID}";

if (!-e "$PATH_LOGS") {
    system("mkdir -p $PATH_LOGS");
}
if (!-e "$PATH_COVERAGE_FILES") {
   system("mkdir -p $PATH_COVERAGE_FILES");
}
if (!-e "$PATH_MUTATION_FILES") {
   system("mkdir -p $PATH_MUTATION_FILES");
}
if (!-e "$PATH_CORRECT_TESTS") {
   system("mkdir -p $PATH_CORRECT_TESTS");
}
if (!-e "$PATH_INSTRUMENTS") {
    system("mkdir -p $PATH_INSTRUMENTS");
}
if (!-e "$PATH_CORRECT_TESTS_TMP") {
    system("mkdir -p $PATH_CORRECT_TESTS_TMP");
}

# Default paths
my $SER_FILE = "cobertura.ser";
my $XML_FILE = "coverage.xml";

my $EXCL_FILE = "exclude.txt";
my $KILL_FILE = "kill.csv";
my $SUMMARY_FILE = "summary.csv";
my $TEST_MAP_FILE = "testMap.csv";

# This are extracted from DB.pm
my $LINES_TOTAL      = "lines_total";
my $LINES_COVERED    = "lines_covered";
my $BRANCHES_TOTAL   = "branches_total";
my $BRANCHES_COVERED = "branches_covered";

my $RESULTS_LOG  = Log::create_log("$PATH_LOGS/${PID}_tests_execution.01_results.log", ">");
my $SUMMARY_LOG  = Log::create_log("$PATH_LOGS/${PID}_tests_execution.02_summary.log", ">");
my $COMPILE_LOG  = Log::create_log("$PATH_LOGS/${PID}_tests_execution.03_compile.log", ">");
my $RUN_LOG      = Log::create_log("$PATH_LOGS/${PID}_tests_execution.04_run.log", ">");
my $MUTATION_LOG = Log::create_log("$PATH_LOGS/${PID}_tests_execution.05_mutation.log", ">");
my $COVERAGE_ANALYSIS_LOG = Log::create_log("$PATH_LOGS/${PID}_tests_execution.06_coverage_analysis.log", ">");



# The mutation operators that should be enabled
# AOR: Arithmetic Operator Replacement
# LOR: Logical Operator Replacement
# SOR: Shift Operator Replacement
# COR: Relational Operator Replacement
# ROR: Increment Operator Replacement
# ORU: Unary Operator Replacement
# LVR: Literal Value Replacement
# STD: Statement Deletion
my @MUT_OPS = ("AOR", "LOR", "SOR", "COR", "ROR", "ORU", "LVR", "STD");
if (defined($MUT_OPS_FILE)) {
    @MUT_OPS = Mutation::parse_mutation_operators($MUT_OPS_FILE);
}


# Agrupamos por project, version, test_id
# Esta agrupación la hacemos solo para obtener la cantidad total tests por método focal únicos a analizar
my %groups;
for my $object (@list_tests) {
    my $group_key = join '|', @{$object}{qw(pid vid vid_num tid)};
    push @{$groups{$group_key}}, $object;
}


#Syntax error: Validate with AST (Abstract Syntax Tree)
#Build error: Validate of compile
#Failing: Compilan pero fallan en la ejecución (fail for incorrect assertions, or wrong expected beavior, e.. the test expects an exception which is not raised)
#Passing: Syntactically correct, compilable and execute without failing
#Correct: Subconjunto de los tests Passing que cubren al menos una línea de la clase focal: Has coverage
#Covered: Subconjunto de los tests Correct que cubren el método focal correcto proporcionado como entrada: Has coverage

my %focal_methods_by_focal_class_and_version = ();

my @list_results = ();
my @list_coverage_by_test_id = ();
my @list_coverage_by_focal_class = ();
my @list_mutants = ();
my @list_bug_detection = ();

my $num_syntax_error = 0;
my $num_syntax_correct = 0;

my $num_compile_error = 0;
my $num_compile_correct = 0;

my $num_execute_error = 0;
my $num_execute_correct = 0;

my $num_failing_by_class = 0; # This is an auxiliary metric
my $num_failing_by_method = 0; # Yhis is an auxiliary metric
my $num_failing_by_class_and_method = 0; # Yhis is an auxiliary metric
my $num_failing = 0;
my $num_passing = 0; # Pass assertions

my $num_correct = 0; # Coverage lines on focal class
my $num_focal_method_covered = 0; # Coverage focal method on focal class


my $coverage_project = $PID;
my $num_lines = 0;
my $num_lines_covered = 0;
my $num_branches = 0;
my $num_branches_covered = 0;
my $num_methods = 0;
my $num_methods_covered = 0;


my $num_mutants_generated = 0;
my $num_mutants_covered = 0;
my $num_mutants_killed = 0;


my $num_buggy_version_executed_tests = 0;
my $num_buggy_version_fail_compiling = 0;
my $num_buggy_version_fail_executing = 0;
my $num_buggy_version_failing_tests = 0;
my $num_buggy_version_bugs_detected = 0;


# Line separator
my $sep = "-"x80;

my $total_test_suites_to_analyze = scalar(@list_tests);
my $total_tests = scalar(%groups);

# Log current time
$SUMMARY_LOG->log_time("Start benchmark execution");
$SUMMARY_LOG->log_msg("- Found " . $total_test_suites_to_analyze . " test archive(s)");
$SUMMARY_LOG->log_msg("\n");

# $MUTATION_LOG->log_time("Start mutation analysis");
# $MUTATION_LOG->log_msg("\n");


# Agrupamos por project, version, test_id
my %groups_by_version;
# Itera sobre cada objeto en la lista de objetos
for my $object_by_version (@list_tests) {
    # Crea una clave de grupo con los valores de 'project' y 'version'
    my $group_key_by_version = join '|', @{$object_by_version}{qw(pid vid vid_num)};

    # Agrega el objeto al grupo correspondiente
    push @{$groups_by_version{$group_key_by_version}}, $object_by_version;
}
my @sorted_keys_by_version = sort compare_keys_by_version keys %groups_by_version;

my $total_versions = scalar(%groups_by_version);


printf ("\n${sep}${sep}\n");
print "Total versions to analize: $total_versions\n";
print "Total focal methods to test: $total_tests\n";
print "Total test suites to analize: $total_test_suites_to_analyze";
printf ("\n${sep}${sep}\n\n");


# Obtenemos el proyecto
my $project = Project::create_project($PID);
$project->{prog_root} = $TMP_DIR;

my $MUTATIONS_TEST_DIR = "$TMP_DIR/bench_mutations_tests";
# my $MUTATIONS_TEST_DIR = "$PATH_LOGS/bench_mutations_tests";

my $count_versions = 1;
my $count_analyzed = 1;
my $prev_vid;


# Itera sobre cada grupo en el hash de grupos
# while (my ($group_key, $list_tests_by_group) = each %groups) {
for my $key_by_version (@sorted_keys_by_version) {
    # Divide la clave del grupo en 'project' y 'version'
    my ($g_project_1, $g_version_1, $g_version_num_1) = split(/\|/, $key_by_version);
    # Ejemplo: $g_project_1=Gson, $g_version_1=9f, $g_version_num_1=9

    my $project_with_version = "${g_project_1}_${g_version_1}";

    printf ("\n\n${sep}${sep}\n");
    print "Inicia Análisis de versión: $project_with_version ($count_versions de $total_versions)";
    printf ("\n${sep}${sep}\n\n");

    $count_versions += 1;
    
    my $list_tests_by_version = $groups_by_version{$key_by_version};

    my $total_test_suites_by_version = scalar(@$list_tests_by_version);

    # if ($g_version_num_1 != 9) {
    #     next;
    # }
    # print "\n\n\n\n\n${sep} \nANALYZE_ALL_TEST_SUITES: $ANALYZE_ALL_TEST_SUITES \n${sep} \n\n\n\n\n";

    # print "\n\nkey_by_version: $key_by_version \n";

    # Agrupamos de nuevo por por project, version, test_id
    my %groups_by_version_and_test_id;
    for my $object_by_version_and_test_id (@$list_tests_by_version) {
        # Crea una clave de grupo con los valores de 'project', 'version' y 'test_id'
        my $group_key_by_version_and_test_id = join '|', @{$object_by_version_and_test_id}{qw(pid vid vid_num tid)};
        push @{$groups_by_version_and_test_id{$group_key_by_version_and_test_id}}, $object_by_version_and_test_id;
    }
    my @sorted_keys_by_version_and_test_id = sort compare_keys_by_version_and_test_id keys %groups_by_version_and_test_id;

    $project->checkout_vid($g_version_1);
    $project->compile() or die "Cannot compile sources for project_version $project_with_version";

    # if (!$prev_vid || $prev_vid ne $g_version_1) {
    #     # Hacemos un git checkout de la versión del proyecto y compilamos
    #     # solo si la versión actual es distinta a la anterior
    #     $project->checkout_vid($g_version_1);
    #     $project->compile() or die "Cannot compile sources for project_version $project_with_version";
    # }
    # $prev_vid = $g_version_1;

    my %focal_methods_by_focal_class = ();


    my %hashes_folders_focal_classes_cobertura = ();
    my $path_bench_cobertura_by_focal_class = "$TMP_DIR/bench_cobertura_by_focal_class";


    my $mutations_files_folder_by_version = "$PATH_MUTATION_FILES/$g_version_1";
    # if (!-e "$mutations_files_folder_by_version") {
    #     system("mkdir -p $mutations_files_folder_by_version");
    # }

    if (!-e "$MUTATIONS_TEST_DIR") {
        # Como este directorio se encuentra dentro de $TMP_DIR, se debe crear después de hacer el checkout
        system("mkdir -p $MUTATIONS_TEST_DIR");
    }
    _clean_mutation_results($TMP_DIR);

    my $should_execute_mutations = 0;
    my %hashes_mutations_instruments_classes = ();
    my $mutations_test_dir_by_version = "$MUTATIONS_TEST_DIR/$g_version_1";
    my $mutations_test_dir_by_version_bak = "$MUTATIONS_TEST_DIR/${g_version_1}_bak";
    if (!-e "$mutations_test_dir_by_version") {
        system("mkdir -p $mutations_test_dir_by_version");
    }
    if (!-e "$mutations_test_dir_by_version_bak") {
        system("mkdir -p $mutations_test_dir_by_version_bak");
    }

    #Build object mut_results -> Los mutantes los ejecutamos a nivel de versión
    my $mut_results = {
        name => $project_with_version,
        project => $g_project_1,
        version => $g_version_1,
        version_num => $g_version_num_1,
        mutants_generated => 0,
        mutants_covered => 0,
        mutants_killed => 0,
        mutation_score_kill_over_cov => 0.0,
        mutation_score_kill_over_gen => 0.0,
        mutation_run_time => 0.0
    };


    my %correct_tests_by_focal_class = ();
    my $should_execute_bug_detection = 0;
    my $tmp_correct_tests_dir_by_version = "$PATH_CORRECT_TESTS_TMP/$g_version_1";
    my $tmp_correct_tests_dir_by_version_bak = "$PATH_CORRECT_TESTS_TMP/${g_version_1}_bak";
    my $correct_tests_dir_by_version = "$PATH_CORRECT_TESTS/$g_version_1";
    if (!-e "$tmp_correct_tests_dir_by_version") {
        system("mkdir -p $tmp_correct_tests_dir_by_version");
    }
    if (!-e "$tmp_correct_tests_dir_by_version_bak") {
        system("mkdir -p $tmp_correct_tests_dir_by_version_bak");
    }
    # if (!-e "$correct_tests_dir_by_version") {
    #     system("mkdir -p $correct_tests_dir_by_version");
    # }

    my $count_test_suites_by_version = 1;


    for my $key_by_version_and_test_id (@sorted_keys_by_version_and_test_id) {
        my ($g_project, $g_version, $g_version_num, $g_test_id) = split(/\|/, $key_by_version_and_test_id);
        # Ejemplo: $g_project=Gson, $g_version=9f, $g_version_num=9, $g_test_id=1

        # print "key_by_version_and_test_id: $key_by_version_and_test_id \n";
        # next;

        my $list_tests_by_version_and_test_id = $groups_by_version_and_test_id{$key_by_version_and_test_id};

        my $name_result = "${g_project}_${g_version}_${g_test_id}";

        #Build object res
        my $res = {
            name => $name_result,
            project => $g_project,
            version => $g_version,
            version_num => $g_version_num,
            test_id => $g_test_id,
            focal_class => "",
            focal_method => "",
            total_suites_analyzed => 0,
            first_suite_id_correct_sintax => undef,
            first_suite_id_compilable => undef,
            first_suite_id_executable => undef,
            first_suite_id_passing_test => undef,
            first_suite_id_correct_test => undef,
            first_suite_id_focal_method_covered => undef,
            correct_syntax => "No",
            compilable => "No",
            executable => "No",
            passing_test => "No",
            correct_test => "No",
            focal_method_covered => "No"
        };

        #Build object cov_result_by_test_id
        my $cov_result_by_test_id = {
            name => $name_result,
            project => $g_project,
            version => $g_version,
            version_num => $g_version_num,
            test_id => $g_test_id,
            instrument_class => "",
            focal_method => "",
            lines_total => 0,
            lines_covered => 0,
            line_coverage => 0.0,
            branches_total => 0,
            branches_covered => 0,
            branch_coverage => 0.0,
            focal_method_covered => "No"
        };

        my $path_bench_cobertura_by_test_id = "$TMP_DIR/bench_cobertura_by_test_id";

        my $coverage_executed = 0;
        my $max_lines_covered = 0;
        my $max_branches_covered = 0;

        my $syntax = 0;
        my $compile = 0;
        my $execute = 0;
        my $passing = 0;
        my $correct = 0;
        my $method_covered = 0;
        my $fail_by_class = 0;
        my $fail_by_method = 0;
        my $fail_by_class_and_method = 0;

        my $count_by_test_id = 0;

        my $should_execute_next_test_suite = 1;

        suite: foreach (@$list_tests_by_version_and_test_id) {
            if (defined $MAX_ATTEMPTS && $count_by_test_id >= $MAX_ATTEMPTS) {
                last suite;
            }
            $count_by_test_id += 1;

            # Obtenemos la info de cada archivo de tests
            my $file_name = $_->{file_name};
            my $pid  = $_->{pid};
            my $vid  = $_->{vid};
            my $src  = $_->{src};
            my $tid  = $_->{tid};
            my $sid  = $_->{sid};

            my $tag_version_tid_sid = "v_${vid} -> tid_${tid}--sid_${sid}";

            if (!$ANALYZE_ALL_TEST_SUITES && !$should_execute_next_test_suite) {
                print "\n\n\nSe Omite test suite $count_analyzed de $total_test_suites_to_analyze ($pid -> $tag_version_tid_sid) ";
                print "-> (Count for this version: $count_test_suites_by_version of $total_test_suites_by_version)\n\n";
                $count_analyzed += 1;
                $count_test_suites_by_version += 1;
                next suite;
            }

            print "\n\n\nAnalizando test suite $count_analyzed de $total_test_suites_to_analyze ($pid -> $tag_version_tid_sid) ";
            print "-> (Count for this version: $count_test_suites_by_version of $total_test_suites_by_version)\n\n";
            $count_analyzed += 1;
            $count_test_suites_by_version += 1;


            $res->{total_suites_analyzed} += 1;

            # if ($tid != 18 && $tid != 23 && $tid != 34) { # Gson 9f
            #     next suite;
            # }

            # Limpiamos los resultados de ejecuciones previas
            Utils::clean_test_results($TMP_DIR);
            # _clean_mutation_results($TMP_DIR);

            printf ("$sep\n$file_name\n$sep\n");


            # project_test_dir es donde descomprimimos el contenido del archivo tar.bz que contiene la clase de test
            my $project_test_dir = "$TMP_DIR/$src";
            # Extract generated tests into temp directory: $file_name is de tar.bz file
            Utils::extract_test_suite("$SUITE_DIR/$file_name", $project_test_dir)
                or die "Cannot extract test suite $file_name";
            
            # En @instruments_to_coverage solo se termina almacenando una clase focal,
            # ya que cada suite corresponde a un único metodo de test para una única clase focal
            my @instruments_to_coverage;
            my $test_java_file = "";
            my $test_java_class = "";
            my $focal_java_class = "";
            my $focal_method_file;
            my $focal_method_name;

            # Buscamos el archivo que contiene el nombre del método focal dentro del archivo comprimido
            find(sub {
                return unless -f; # Retorna si no es un file
                return unless /\.txt$/; # Retorna si no es un archivo.txt
                # my $txt_file_find = basename($File::Find::name, '.txt'); # Sin sufijo
                my $txt_file_find = basename($File::Find::name);
                # print(STDERR "File finding: $txt_file_find\n");

                if ($txt_file_find eq "focal_method_name.txt") {
                    $focal_method_file = $File::Find::name;
                    # print(STDERR "File with focal_method_name: $focal_method_file\n");
                }
            }, $project_test_dir);

            if ($focal_method_file) {
                open(my $fh, '<', $focal_method_file) or die "No se pudo abrir el archivo '$focal_method_file' $!";
                $focal_method_name = <$fh>;
                print "\nFocal_method_name to test: $focal_method_name\n";
                close($fh);
            }


            # Buscamos la clase de test dentro del archivo comprimido
            find(sub {
                # if (-f && /\.java$/) {
                return unless -f; # Retorna si no es un file
                return unless /\.java$/; # Retorna si no es un archivo.java
                if (!/scaffolding\.java$/) { # Obtenemos los que no terminan en scaffolding.java
                    my $java_file_find = $File::Find::name;
                    $test_java_file = $java_file_find;
                    # Eliminamos la parte de la ruta base para obtener la ruta relativa
                    $java_file_find =~ s/^$project_test_dir\/?//; # Only com/google/gson/MyClass_ESTests.java
                    $java_file_find =~ s/\//./g;              # Reemplaza '/' por '.'
                    # $java_file_find =~ s/\/\././g;
                    $test_java_class = $java_file_find;
                    $test_java_class =~ s/\.java$//;
                    if ($USE_SCAFFOLDING) {
                        $java_file_find =~ s/_ESTest\.java$//;  # Elimina '_ESTest.java' al final de la cadena
                    } else {
                        $java_file_find =~ s/Test\.java$//;     # Elimina 'Test.java' al final de la cadena
                    }
                    $focal_java_class = $java_file_find;
                    push @instruments_to_coverage, $java_file_find;
                }
            }, $project_test_dir);

            my $file_instrument_classes = "${PATH_INSTRUMENTS}/${pid}_${vid}_instruments_${tid}_${sid}.txt";
            open my $fh_ins, '>', $file_instrument_classes or die "No se pudo abrir el archivo $file_instrument_classes: $!";
            print $fh_ins @instruments_to_coverage;
            close $fh_ins;

            # Set instrument class in cov_result_by_test_id
            $res->{focal_class} = $focal_java_class;
            $cov_result_by_test_id->{instrument_class} = $focal_java_class;

            if ($focal_method_name) {
                $res->{focal_method} = $focal_method_name;
                $cov_result_by_test_id->{focal_method} = $focal_method_name;
            }

            if (exists $focal_methods_by_focal_class{$focal_java_class}) {
                if (!grep { $_ eq $focal_method_name } @{$focal_methods_by_focal_class{$focal_java_class}}) {
                    # Solo lo agregamos si no existe en la lista
                    push(@{$focal_methods_by_focal_class{$focal_java_class}}, $focal_method_name);
                }
                # push(@{$focal_methods_by_focal_class{$focal_java_class}}, $focal_method_name);
            } else {
                $focal_methods_by_focal_class{$focal_java_class} = [$focal_method_name];
            }


            printf ("\n Test Class to analyze: $test_java_file\n\n");
            my $has_syntax_error = _get_result_parser_utils($test_java_file, "HAS_ERROR"); # Error | No Error
            my $test_method_name = _get_result_parser_utils($test_java_file, "GET_METHOD_NAME");

            my $test_canonical_name = "${test_java_class}::$test_method_name";

            print "Resultado has_syntax_error ($tag_version_tid_sid): $has_syntax_error\n";
            print "Resultado test_method_name ($tag_version_tid_sid): $test_method_name\n";
            print "Resultado test_canonical_name ($tag_version_tid_sid): $test_canonical_name\n\n";


            # Validate Syntax error
            if ($has_syntax_error eq "Error") {
                $SUMMARY_LOG->log_msg(" - Syntax error in test method(s): $file_name");
                $SUMMARY_LOG->log_msg("   --- $test_canonical_name");
                $SUMMARY_LOG->log_msg("");

                print "\n\n";
                template_validation_property("There is error syntaxt error in test suite $tag_version_tid_sid!");

                next suite;
            }
            if (!$syntax) {
                $num_syntax_correct += 1;
                $res->{first_suite_id_correct_sintax} = $sid;
                $res->{correct_syntax} = "Yes";
                $syntax = 1;
            }
            print "CORRECT SYNTAX for test ($tag_version_tid_sid): $test_canonical_name\n\n";


            # Temporary log file to monitor uncompilable tests
            # Con el modo ">", siempre se sobreescribe el archivo anterior
            my $compilation_log = Log::create_log("$TMP_DIR/comp_tests.log", ">")->{file_name};

            # Check for compilation errors
            if (! $project->compile_ext_tests($project_test_dir, $compilation_log)) {
                #$COMPILE_LOG->log_file("=> Compilation issues: $file_name", $compilation_log);
                $COMPILE_LOG->log_msg("=> Compilation issues: $file_name\n");
                open(my $log_fh, '<', $compilation_log) or die "No se pudo abrir el archivo $compilation_log: $!";
                while (my $line = <$log_fh>) {
                    # Omitimos líneas que comienzan con:
                    # [javac] warning:
                    # [javac] It is recommended that the compiler be upgraded
                    # [javac] Creating empty
                    # [copy] Copying
                    if ($line =~ /javac.*warning:/ 
                            || $line =~ /javac.*It is recommended that the compiler be upgraded/
                            || $line =~ /javac.*Creating empty/
                            || $line =~ /copy.*Copying/) {
                        next;
                    }
                    chomp($line); # Usamos chomp para eliminar el salto de línea al final (\n)
                    $COMPILE_LOG->log_msg($line);
                }

                close($log_fh);
                $COMPILE_LOG->log_msg("\n\n\n\n\n");

                my ($n_uncompilable_tests, $n_uncompilable_test_classes) = _rm_classes($compilation_log, $src, $file_name, $test_method_name);

                print "\n\n";
                template_validation_property("Cannot compile test suite $tag_version_tid_sid");

                next suite;
            }
            if (!$compile) {
                $num_compile_correct += 1;
                $res->{first_suite_id_compilable} = $sid;
                $res->{compilable} = "Yes";
                $compile = 1;
            }
            print "COMPILABLE for test ($tag_version_tid_sid): $test_canonical_name\n\n";


            # Instrument all classes provided before run_ext_tests
            $project->coverage_instrument($file_instrument_classes) or die "Cannot coverage_instrument for test suite $tag_version_tid_sid";


            # Temporary log file to monitor failing tests
            my $log_run_tests = Log::create_log("$TMP_DIR/run_tests.log", ">")->{file_name};

            if (! $project->run_ext_tests($project_test_dir, "$INCL", $log_run_tests)) {
                # Esto lo guardamos en SUMMARY_LOG y no en RUN_LOG, ya que si falla la ejecución
                # probablemente es debido a un problema con ant
                $SUMMARY_LOG->log_file(" - Test not executable: $file_name", $log_run_tests);

                print "\n\n";
                template_validation_property("Cannot run test suite $tag_version_tid_sid");

                next suite;
            }
            if (!$execute) {
                $num_execute_correct += 1;
                $res->{first_suite_id_executable} = $sid;
                $res->{executable} = "Yes";
                $execute = 1;
            }
            print "EXECUTABLE for test ($tag_version_tid_sid): $test_canonical_name\n\n";


            # Check failing test classes and methods after run_ext_tests
            my $passing_actual_test_suite = 1;
            my $list_failings = Utils::get_failing_tests($log_run_tests) or die;
            if (scalar(@{$list_failings->{classes}}) != 0 || scalar(@{$list_failings->{methods}}) != 0) {
                my $has_failing_class = 0;

                if (scalar(@{$list_failings->{classes}}) != 0) {
                    $has_failing_class = 1;
                    $RUN_LOG->log_file("Failing test class: $file_name", $log_run_tests);
                    $RUN_LOG->log_msg("");

                    $SUMMARY_LOG->log_msg(" - Failing test class: $file_name");
                    $SUMMARY_LOG->log_msg(join("\n", @{$list_failings->{classes}}));
                    $SUMMARY_LOG->log_file(" - Stack traces:", $log_run_tests);
                    #$SUMMARY_LOG->log_msg("");

                    #$num_failing_by_class += 1;
                    $fail_by_class = 1;
                    print "FAILING TEST (by class) for test ($tag_version_tid_sid): $test_canonical_name\n";
                }

                if (scalar(@{$list_failings->{methods}}) != 0) {
                    $RUN_LOG->log_file(scalar(@{$list_failings->{methods}}) . " broken test method(s): $file_name", $log_run_tests);
                    $RUN_LOG->log_msg("");

                    my $msg_prefix = " - ";
                    if ($has_failing_class) {
                        #$num_failing_by_class_and_method += 1;
                        $fail_by_class_and_method = 1;
                        $msg_prefix = " --> And ";
                    }
                    $SUMMARY_LOG->log_msg("${msg_prefix}Finding " . scalar(@{$list_failings->{methods}}) . " broken test method(s): $file_name");
                    $SUMMARY_LOG->log_msg(join("\n", @{$list_failings->{methods}}));
                    #$SUMMARY_LOG->log_msg("");
                    
                    #$num_failing_by_method += 1;
                    $fail_by_method = 1;
                    print "FAILING TEST (by method) for test ($tag_version_tid_sid): $test_canonical_name\n";

                    # push(@list_results, $res);
                }

                $RUN_LOG->log_msg("\n");
                $SUMMARY_LOG->log_msg("");

                print "\n\n";
                template_validation_property("Failing test for test suite $tag_version_tid_sid");

                $passing_actual_test_suite = 0;
                # next suite;
            }
            # if (!$passing) {
            if ($passing_actual_test_suite) {
                if (!$passing) {
                    $num_passing += 1;
                    $res->{first_suite_id_passing_test} = $sid;
                    $res->{passing_test} = "Yes";
                    $passing = 1;
                }
                print "PASSING ASSERTS for test ($tag_version_tid_sid): $test_canonical_name\n\n";

                $SUMMARY_LOG->log_msg(" - PASSED CORRECTLY ASSERTS test method(s): $file_name");
                $SUMMARY_LOG->log_msg("$test_canonical_name");
                $SUMMARY_LOG->log_msg("");
            }
            

            # if ($VALIDATE_TEMPLATE) {
            #     last suite;
            # }
            

            # El coverage lo ejecutamos solo sobre los tests que compilan y que se ejecutan
            # independientemente que pasen o no los asserts

            # Obtenemos el src_dir, ya que pueden haber clases con el mismo package en otras carpetas del proyecto
            my $src_dir = $project->src_dir($vid);
            # my $SINGLE_TEST = $test_canonical_name;

            # Generate coverage report. Coverage report was generated in run_ext_tests
            $project->coverage_report($src_dir) or die "Could not create coverage report for test suite $tag_version_tid_sid";
            $coverage_executed = 1;

            # Parse xml output and return coverage ratios
            my $xmlfile  = "$project->{prog_root}/$XML_FILE";
            my $cov_info = _get_info_from_xml($xmlfile);


            # print "Focal class: $focal_java_class \n";
            my $focal_java_class_folder_coverage = $focal_java_class;
            $focal_java_class_folder_coverage =~ s/\./\_/g; # Reemplaza '.' por '_'
            # print "Focal class underscore: $focal_java_class_folder_coverage \n";

            # Copiamos el archivo coverage.xml en la ruta de resultados
            my $path_coverage_files_by_version_and_focal_class = "$PATH_COVERAGE_FILES/${vid}/${focal_java_class_folder_coverage}";
            # print "Focal class folder: $path_coverage_files_by_version_and_focal_class \n";
            if (!-e "$path_coverage_files_by_version_and_focal_class") {
                # system("mkdir -p $path_coverage_files_by_version_and_focal_class");
                # Si no se escapan los caracteres especiales, por ejemplo com.google.gson.$Gson$Types, quedaría com_google_gson_
                my $path_coverage_files_by_version_and_focal_class_escaped = quotemeta($path_coverage_files_by_version_and_focal_class);
                system("mkdir -p $path_coverage_files_by_version_and_focal_class_escaped");
                # make_path($destino_completo);
            }

            my $xml_coverage = "$path_coverage_files_by_version_and_focal_class/coverage_${pid}_${vid}_${tid}_s${sid}.xml";
            copy($xmlfile, $xml_coverage) or die "La copia del archivo $xml_coverage falló: $!";

            
            my $serfile  = "$project->{prog_root}/$SER_FILE";

            my $path_test_id_cobertura = "$path_bench_cobertura_by_test_id/$vid/$tid";
            if (!-e "$path_test_id_cobertura") {
                my $path_test_id_cobertura_escaped = quotemeta($path_test_id_cobertura);
                system("mkdir -p $path_test_id_cobertura_escaped");
                # system("mkdir -p $path_test_id_cobertura");
            }
            my $cobertura_ser_test_id = "$path_test_id_cobertura/cobertura_${pid}_${vid}_${tid}_s${sid}.ser";
            copy($serfile, $cobertura_ser_test_id) or die "La copia del archivo $cobertura_ser_test_id falló: $!";

            my $path_focal_class_cobertura = "$path_bench_cobertura_by_focal_class/$focal_java_class_folder_coverage";
            if (!-e "$path_focal_class_cobertura") {
                my $path_focal_class_cobertura_escaped = quotemeta($path_focal_class_cobertura);
                system("mkdir -p $path_focal_class_cobertura_escaped");
                # system("mkdir -p $path_focal_class_cobertura");
            }
            my $cobertura_ser_class = "$path_focal_class_cobertura/cobertura_${pid}_${vid}_${tid}_s${sid}.ser";
            copy($serfile, $cobertura_ser_class) or die "La copia del archivo $cobertura_ser_class falló: $!";
            $hashes_folders_focal_classes_cobertura{$focal_java_class_folder_coverage} = $focal_java_class;


            if ($cov_info->{lines_covered} > 0) {
                if (!$correct) {
                    $num_correct += 1;
                    $res->{first_suite_id_correct_test} = $sid;
                    $res->{correct_test} = "Yes";
                    $correct = 1;
                }

                print "CORRECT TEST (LINE COVERED) ($tag_version_tid_sid): $test_canonical_name\n\n";
            }

            $cov_result_by_test_id->{lines_total}    = $cov_info->{lines_total};
            $cov_result_by_test_id->{branches_total} = $cov_info->{branches_total};

            # if ($cov_info->{lines_covered} > $max_lines_covered) {
            #     $max_lines_covered = $cov_info->{lines_covered};
            #     $cov_result_by_test_id->{lines_covered} = $cov_info->{lines_covered};
            #     $cov_result_by_test_id->{line_coverage} = $cov_info->{lines_covered}/$cov_info->{lines_total}*100;
            # }

            # if ($cov_info->{branches_covered} > $max_branches_covered) {
            #     $max_branches_covered = $cov_info->{branches_covered};
            #     $cov_result_by_test_id->{branches_covered} = $cov_info->{branches_covered};
            #     $cov_result_by_test_id->{branch_coverage}  = ($cov_info->{branches_total} == 0 ? 0 : $cov_info->{branches_covered}/$cov_info->{branches_total}*100);
            # }


            my $suite_id_method_covered = 0;
            if ($cov_info->{lines_covered} > 0) {
                if ($focal_method_name && -e $xml_coverage) {
                    my $coverage_method_result = _validate_method_coverage($focal_method_name, $xml_coverage);
                    if ($coverage_method_result eq "COVERED") {
                        $suite_id_method_covered = 1;
                        if (!$method_covered) {
                            $num_focal_method_covered += 1;
                            $res->{first_suite_id_focal_method_covered} = $sid;
                            $res->{focal_method_covered} = "Yes";
                            $cov_result_by_test_id->{focal_method_covered} = "Yes";
                            $method_covered = 1;
                        }

                        print "FOCAL METHOD $focal_method_name COVERED ($tag_version_tid_sid): $test_canonical_name\n\n";
                    }
                } else {
                    $res->{focal_method_covered} = "Undefined";
                    $cov_result_by_test_id->{focal_method_covered} = "Undefined";
                }
            }

            # if ($VALIDATE_TEMPLATE) {
            #     last suite;
            # }


            # El análisis de mutantes solo lo ejecutamos sobre los tests que pasan correctamente
            # y que cubren al menos una línea de la clase focal (es decir sobre tests correctos)
            # Sin embargo, los mutantes los ejecutamos al final de cada versión, por lo tanto,
            # a continuación solo extraemos los tests sobre los cuales se ejecutará dicho análisis
            # if (!($focal_java_class =~ /\$/) && $passing_actual_test_suite) {
            if (($VALIDATE_TEMPLATE && !($focal_java_class =~ /\$/) && $passing_actual_test_suite)
                || (!($focal_java_class =~ /\$/) && $passing_actual_test_suite && $cov_info->{lines_covered} > 0)) {
            
                my $mutation_test_src = "$mutations_test_dir_by_version_bak/$src";

                # 9f_bak/generated_test_xx
                if (!-e "$mutation_test_src") {
                    system("mkdir -p $mutation_test_src");
                }

                # 9f_bak/generated_test_xx/com/some/SomeTest.java
                copy_directory($project_test_dir, $mutation_test_src);

                # Renombramos las clases de test, ya que de lo contrario, al volver a compilarlas y ejecutarlas
                # en la nueva carpeta, tendríamos un error por nombres de clases duplicadas
                my $file_mutation_java_test = find_file_java_test($mutation_test_src, $focal_java_class);
                my $new_file_mutation_java_test = $file_mutation_java_test;
                $new_file_mutation_java_test =~ s/\.java$//;
                $new_file_mutation_java_test = "${new_file_mutation_java_test}_v${vid}_t${tid}_s${sid}.java";

                my $actual_test_class_name = basename($file_mutation_java_test, '.java');
                my $new_test_class_name = "${actual_test_class_name}_v${vid}_t${tid}_s${sid}";

                _update_test_class_name($actual_test_class_name, $new_test_class_name, $file_mutation_java_test, $new_file_mutation_java_test);

                # 9f/com/some/SomeTest_xx.java
                copy_directory_and_only_java_files($mutation_test_src, $mutations_test_dir_by_version);

                $hashes_mutations_instruments_classes{$focal_java_class} = 1; # Los hashes no permiten claves duplicadas
                $should_execute_mutations = 1;
            }


            # El análisis de bug detection solo lo ejecutamos sobre los tests que pasan correctamente
            # y que cubren al menos una línea de la clase focal (es decir, sobre tests correctos)
            if (($VALIDATE_TEMPLATE && $passing_actual_test_suite)
                || ($passing_actual_test_suite && $cov_info->{lines_covered} > 0)) {

                my $correct_test_src = "$correct_tests_dir_by_version/$src";
                if (!-e "$correct_test_src") {
                    system("mkdir -p $correct_test_src");
                }
                copy_directory($project_test_dir, $correct_test_src);

                
                my $tmp_correct_test_src = "$tmp_correct_tests_dir_by_version_bak/$src";
                # 9f_bak/generated_test_xx
                if (!-e "$tmp_correct_test_src") {
                    system("mkdir -p $tmp_correct_test_src");
                }
                # 9f_bak/generated_test_xx/com/some/SomeTest.java
                copy_directory($project_test_dir, $tmp_correct_test_src);

                # Renombramos las clases de test, ya que de lo contrario, al volver a compilarlas y ejecutarlas
                # en la nueva carpeta, tendríamos un error por nombres de clases duplicadas
                my $file_correct_java_test = find_file_java_test($tmp_correct_test_src, $focal_java_class);
                my $new_file_correct_java_test = $file_correct_java_test;
                $new_file_correct_java_test =~ s/\.java$//;
                $new_file_correct_java_test = "${new_file_correct_java_test}_v${vid}_t${tid}_s${sid}.java";

                my $actual_correct_test_class_name = basename($file_correct_java_test, '.java');
                my $new_correct_test_class_name = "${actual_correct_test_class_name}_v${vid}_t${tid}_s${sid}";

                _update_test_class_name($actual_correct_test_class_name, $new_correct_test_class_name, $file_correct_java_test, $new_file_correct_java_test);


                my $focal_java_class_folder_corrects = $focal_java_class;
                $focal_java_class_folder_corrects =~ s/\./\_/g; # Reemplaza '.' por '_'
                my $tmp_correct_tests_dir_by_version_and_focal_class = "$tmp_correct_tests_dir_by_version/${focal_java_class_folder_corrects}";

                if (!-e "$tmp_correct_tests_dir_by_version_and_focal_class") {
                    # system("mkdir -p $path_coverage_files_by_version_and_focal_class");
                    # Si no se escapan los caracteres especiales, por ejemplo com.google.gson.$Gson$Types, quedaría com_google_gson_
                    my $tmp_correct_tests_dir_by_version_and_focal_class_escaped = quotemeta($tmp_correct_tests_dir_by_version_and_focal_class);
                    system("mkdir -p $tmp_correct_tests_dir_by_version_and_focal_class_escaped");
                    # make_path($destino_completo);
                }

                # 9f/com_some_Some/com/some/SomeTest_xx.java
                copy_directory_and_only_java_files($tmp_correct_test_src, $tmp_correct_tests_dir_by_version_and_focal_class);

                my $new_correct_focal_method_file = "$tmp_correct_tests_dir_by_version_and_focal_class/${new_correct_test_class_name}_focal_method_name.txt";
                copy($focal_method_file, $new_correct_focal_method_file) or die "La copia del archivo $new_correct_focal_method_file falló: $!";

                my $metadata_correct_test = {
                    focal_method_name => $focal_method_name,
                    focal_method_covered => $suite_id_method_covered == 1 ? "Yes" : "No",
                    test_suite_src => $src,
                    test_id => $tid,
                    suite_id => $sid,
                    actual_correct_test_class_name => $actual_correct_test_class_name,
                    new_correct_test_class_name => $new_correct_test_class_name
                };

                if (exists $correct_tests_by_focal_class{$focal_java_class_folder_corrects}) {
                    $correct_tests_by_focal_class{$focal_java_class_folder_corrects}->{total_correct_tests} += 1;
                    push(
                        @{$correct_tests_by_focal_class{$focal_java_class_folder_corrects}->{list_metadata}}, 
                        $metadata_correct_test
                    );
                } else {
                    $correct_tests_by_focal_class{$focal_java_class_folder_corrects} = {
                        focal_class => $focal_java_class,
                        total_correct_tests => 1,
                        list_metadata => [$metadata_correct_test]
                    };
                }

                # $correct_tests_by_focal_class{$focal_java_class_folder_corrects} = $focal_java_class;
                
                $should_execute_bug_detection = 1;
            }


            print "\n\n";
            unlink $file_instrument_classes or die "No se pudo eliminar el archivo $file_instrument_classes: $!\n";

            if ($method_covered) {
                $should_execute_next_test_suite = 0;
                # last;
                # last suite;
            }

        }

        if (!$syntax) {
            $num_syntax_error += 1;
        }
        if (!$compile) {
            $num_compile_error += 1;
        }
        if (!$execute) {
            $num_execute_error += 1;
        }
        if (!$passing) {
            $num_failing += 1;
            if ($fail_by_class) {
                $num_failing_by_class += 1;
            }
            if ($fail_by_method) {
                $num_failing_by_method += 1;
            }
            if ($fail_by_class_and_method) {
                $num_failing_by_class_and_method += 1;
            }
        }

        push(@list_results, $res);


        if ($coverage_executed) {
            # cobertura.ser.src.dir: Donde estan los .ser que voy a mergear
            # cobertura.ser.merged.file: Archivo (con path) donde se mergeará  en un solo .ser
            # coverage.report.merged.src.dir: ruta donde se va a guardar el nuevo archivo .xml
            # coverage.src.dir: path donde estan los codigos fuentes
            print "Cobertura Merge for $project_with_version - test_id -> ${g_test_id}:\n";

            my $src_dir = $project->src_dir($g_version_1);

            my $path_test_id_cobertura_files = "$path_bench_cobertura_by_test_id/${g_version_1}/${g_test_id}";
            my $file_test_id_cobertura_merged = "$path_test_id_cobertura_files/cobertura_merged_${g_project_1}_${g_version_1}_${g_test_id}.ser";
            $project->_ant_call_comp("cobertura.merge", "-Dcobertura.ser.src.dir=$path_test_id_cobertura_files " . 
                                                        "-Dcobertura.ser.merged.file=$file_test_id_cobertura_merged " .
                                                        "-Dcoverage.report.merged.src.dir=$path_test_id_cobertura_files " .
                                                        "-Dcoverage.src.dir=$src_dir");
            
            my $test_id_coverage_xml_file = "$path_test_id_cobertura_files/$XML_FILE";
            my $test_id_cov_report = _get_info_from_xml($test_id_coverage_xml_file);
            
            # die "Merge finalizado";

            my $res_coverage_analysis_by_test_id = _run_analyze_coverage_by_d4j_project("TestId", $g_test_id, undef, undef);
            if ($res_coverage_analysis_by_test_id ne "Error") {
                (
                    $cov_result_by_test_id->{lines_total},
                    $cov_result_by_test_id->{branches_total},
                    $cov_result_by_test_id->{lines_covered},
                    $cov_result_by_test_id->{line_coverage},
                    $cov_result_by_test_id->{branches_covered},
                    $cov_result_by_test_id->{branch_coverage}
                ) = split(/\|/, $res_coverage_analysis_by_test_id);
            }

            if ($res_coverage_analysis_by_test_id eq "Error" || $test_id_cov_report->{lines_covered} >= $cov_result_by_test_id->{lines_covered}) {

                $cov_result_by_test_id->{lines_total} = $test_id_cov_report->{lines_total};
                $cov_result_by_test_id->{branches_total} = $test_id_cov_report->{branches_total};
                $cov_result_by_test_id->{lines_covered} = $test_id_cov_report->{lines_covered};
                $cov_result_by_test_id->{line_coverage}  = 
                        $test_id_cov_report->{lines_total} == 0 
                                ? 0.0 
                                : $test_id_cov_report->{lines_covered}/$test_id_cov_report->{lines_total}*100;
                $cov_result_by_test_id->{branches_covered} = $test_id_cov_report->{branches_covered};
                $cov_result_by_test_id->{branch_coverage}  = 
                        $test_id_cov_report->{branches_total} == 0 
                                ? 0.0 
                                : $test_id_cov_report->{branches_covered}/$test_id_cov_report->{branches_total}*100;
            }

            push(@list_coverage_by_test_id, $cov_result_by_test_id);
        }
    }

    # printf ("\n\n focal_methods_by_focal_class\n\n");
    # print Dumper(\%focal_methods_by_focal_class);
    # print "\n\n";

    $focal_methods_by_focal_class_and_version{$g_version_1} = \%focal_methods_by_focal_class;
    # print "\n\n";
    

    my $src_dir = $project->src_dir($g_version_1);

    # foreach my $folder_focal_class (@list_folders_focal_classes_cobertura) {
    while (my ($folder_focal_class, $focal_class_with_package) = each %hashes_folders_focal_classes_cobertura) {
        print "\n\n\nCOVERAGE ANALYSIS BY FOCAL CLASS for $project_with_version: $focal_class_with_package\n\n";

        my $path_focal_class_cobertura_files = "$path_bench_cobertura_by_focal_class/$folder_focal_class";
        my $path_focal_class_cobertura_files_escaped = quotemeta($path_focal_class_cobertura_files);
        my $file_focal_class_cobertura_merged = "$path_focal_class_cobertura_files/cobertura_merged_${g_project_1}_${g_version_1}-${folder_focal_class}.ser";
        my $file_focal_class_cobertura_merged_escaped = quotemeta($file_focal_class_cobertura_merged);
        # my $file_focal_class_cobertura_merged = "$path_focal_class_cobertura_files/cobertura_merged_${g_project_1}_${g_version_1}.ser";

        $project->_ant_call_comp("cobertura.merge", "-Dcobertura.ser.src.dir=$path_focal_class_cobertura_files_escaped " . 
                                                    "-Dcobertura.ser.merged.file=$file_focal_class_cobertura_merged_escaped " .
                                                    "-Dcoverage.report.merged.src.dir=$path_focal_class_cobertura_files_escaped " .
                                                    "-Dcoverage.src.dir=$src_dir");
        
        my $focal_class_coverage_xml_file = "$path_focal_class_cobertura_files/$XML_FILE";
        my $focal_class_cov_report = _get_info_from_xml($focal_class_coverage_xml_file);

        # my $path_coverage_files_by_version_and_focal_class = "$PATH_COVERAGE_FILES/${g_version_1}/${folder_focal_class}";
        my $path_coverage_files_by_version_and_focal_class = "$PATH_COVERAGE_FILES/${g_version_1}";
        my $xml_coverage_merged = "$path_coverage_files_by_version_and_focal_class/coverage_merged_${g_project_1}_${g_version_1}-${folder_focal_class}.xml";
        copy($focal_class_coverage_xml_file, $xml_coverage_merged) or die "La copia del archivo $xml_coverage_merged falló: $!";

        # my $focal_class_with_package = $folder_focal_class;
        # $focal_class_with_package =~ s/\_/\./g; # Reemplaza '_' por '.'

        my $methods_total_of_focal_class = scalar(@{$focal_methods_by_focal_class{$focal_class_with_package}});
        my $methods_covered_of_focal_class = 0;
        my $method_coverage_of_focal_class = 0;
        
        my $list_focal_methods_of_focal_class = join('|', @{$focal_methods_by_focal_class{$focal_class_with_package}});
        my $res_coverage_analysis_by_focal_class = _run_analyze_coverage_by_d4j_project(
            "GetMethodCoverage", undef, $focal_class_coverage_xml_file, $list_focal_methods_of_focal_class
        );

        if ($res_coverage_analysis_by_focal_class ne "Error") {
            (
                $methods_covered_of_focal_class,
                $method_coverage_of_focal_class
            ) = split(/\|/, $res_coverage_analysis_by_focal_class);
        }

        my $cov_result_by_focal_class = {
            name => $project_with_version,
            project => $g_project_1,
            version => $g_version_1,
            version_num => $g_version_num_1,
            instrument_class => $focal_class_with_package,
            lines_total => $focal_class_cov_report->{lines_total},
            lines_covered => $focal_class_cov_report->{lines_covered},
            line_coverage => $focal_class_cov_report->{lines_total} == 0 
                                ? 0.0 
                                : $focal_class_cov_report->{lines_covered}/$focal_class_cov_report->{lines_total}*100,
            branches_total => $focal_class_cov_report->{branches_total},
            branches_covered => $focal_class_cov_report->{branches_covered},
            branch_coverage => $focal_class_cov_report->{branches_total} == 0 
                                ? 0.0 
                                : $focal_class_cov_report->{branches_covered}/$focal_class_cov_report->{branches_total}*100,
            methods_total => $methods_total_of_focal_class,
            methods_covered => $methods_covered_of_focal_class,
            method_coverage => $method_coverage_of_focal_class
        };

        $num_lines += $focal_class_cov_report->{lines_total};
        $num_lines_covered += $focal_class_cov_report->{lines_covered};
        $num_branches += $focal_class_cov_report->{branches_total};
        $num_branches_covered += $focal_class_cov_report->{branches_covered};
        $num_methods += $methods_total_of_focal_class;
        $num_methods_covered += $methods_covered_of_focal_class;

        push(@list_coverage_by_focal_class, $cov_result_by_focal_class);
        # print "\n\n";
    }
    # die "Finaliza merge by class";


    if ($should_execute_mutations) {
        print "\n\n\nMUTATION ANALYSIS for $project_with_version:\n\n";

        my @mutations_instruments_classes = keys %hashes_mutations_instruments_classes;

        my $file_mutations_instruments_classes = "${mutations_test_dir_by_version}/${g_project_1}_${g_version_1}_instruments.txt";
        open(my $fh_mut_ins, '>', $file_mutations_instruments_classes) or die "No se pudo abrir el archivo $file_mutations_instruments_classes: $!";
        foreach my $instrument_class (@mutations_instruments_classes) {
            print $fh_mut_ins "$instrument_class\n";
        }
        close($fh_mut_ins);


        if ($project->mutate($file_mutations_instruments_classes, \@MUT_OPS) > 0) {
            # Si se generaron mutantes, procedemos a ejecutar el análisis

            # Ya habíamos compilado previamente los tests, sin embargo,
            # como los copiamos en otra carpeta, los volvemos a compilar
            $project->compile_ext_tests($mutations_test_dir_by_version) or die "Mutation Tests do not compile!";
            
            # Temporary log file to monitor mutation analysis
            #my $mut_log = "$TMP_DIR/.mutation.log"; `>$mut_log`;
            my $log_mutation = Log::create_log("$TMP_DIR/.mutation.log", ">")->{file_name};

            # No need to run the test suite first. Major's preprocessing verifies that
            # all tests in the test suite pass before performing the mutation analysis.

            # my $mut_map = Mutation::mutation_analysis_ext($project, $project_test_dir, "$INCL", $mut_log, $EXCL);
            # my $mut_map = Mutation::mutation_analysis_ext($project, $project_test_dir, "$INCL", $log_mutation, undef);

            # my $mut_result = $project->mutation_analysis_ext($project_test_dir, "$INCL", $log_file, $EXCL, $SINGLE_TEST);
            # my $mut_result = $project->mutation_analysis_ext($mutations_test_dir_by_version, "$INCL", $log_mutation, undef, undef);
            my $mut_result = _mutation_analysis_ext($project, $mutations_test_dir_by_version, "$INCL", $log_mutation, undef, undef);

            if (! $mut_result) {
                # system("cat $log_file");
                # $MUTATION_LOG->log_file("Failing mutation analysis: $project_with_version", $log_mutation);
                $MUTATION_LOG->log_msg("=> Failing mutation analysis: $project_with_version\n");
                open(my $log_fh, '<', $log_mutation) or die "No se pudo abrir el archivo $log_mutation: $!";
                while (my $line = <$log_fh>) {
                    # Omitimos líneas que comienzan con:
                    # [javac] warning:
                    # [javac] It is recommended that the compiler be upgraded
                    # [javac] Creating empty
                    # [copy] Copying
                    if ($line =~ /javac.*warning:/ 
                            || $line =~ /javac.*It is recommended that the compiler be upgraded/
                            || $line =~ /javac.*Creating empty/
                            || $line =~ /copy.*Copying/) {
                        next;
                    }
                    chomp($line); # Usamos chomp para eliminar el salto de línea al final (\n)
                    $MUTATION_LOG->log_msg($line);
                }
                close($log_fh);
                $MUTATION_LOG->log_msg("\n\n\n\n\n");

                template_validation_property("Failing mutation analysis for version $project_with_version");

                next;
            }

            open(RES, "<$TMP_DIR/summary.csv") or die "Cannot read mutation analysis result file for project_version '$project_with_version' $!";
            # open(RES, "<$TMP_DIR/summary.csv") or next;
            my $header = <RES>;
            my $line = <RES>;
            close(RES);
            
            my ($mutGen, $mutCov, $mutKill, $mutLive, $timePreproc, $timeAnalysis) = split(/,/, $line);
            my $killRatioGen = $mutGen == 0 ? 0.0 : $mutKill/$mutGen*100;
            my $killRatioCov = $mutCov == 0 ? 0.0 : $mutKill/$mutCov*100;
            # print "\n\n\n\n\n";
            # printf("%17s: %d\n", "Mutants generated", $mutGen);
            # printf("%17s: %d\n", "Mutants covered", $mutCov);
            # printf("%17s: %d\n", "Mutants killed", $mutKill);
            # printf("%17s: %.1f%% (%.1f%%)\n", "Mutation score", $killRatioCov, $killRatioGen);
            # printf("%17s: %.1f seconds\n", "Total run time", ($timePreproc+$timeAnalysis));
            # print "\n\n\n\n\n";

            $mut_results->{mutants_generated} = $mutGen;
            $mut_results->{mutants_covered} = $mutCov;
            $mut_results->{mutants_killed} = $mutKill;
            $mut_results->{mutation_score_kill_over_cov} = $killRatioCov;
            $mut_results->{mutation_score_kill_over_gen} = $killRatioGen;
            $mut_results->{mutation_run_time} = ($timePreproc+$timeAnalysis);

            $num_mutants_generated += $mutGen;
            $num_mutants_covered += $mutCov;
            $num_mutants_killed += $mutKill;

            push(@list_mutants, $mut_results);

            if (!-e "$mutations_files_folder_by_version") {
                system("mkdir -p $mutations_files_folder_by_version");
            }

            my $summary_mutation_file = "$mutations_files_folder_by_version/summary_${g_project_1}_${g_version_1}.csv";
            if (-e "$TMP_DIR/summary.csv") {
                copy("$TMP_DIR/summary.csv", $summary_mutation_file) or die "La copia del archivo $summary_mutation_file falló: $!";
            }

            my $kill_mutation_file = "$mutations_files_folder_by_version/kill_${g_project_1}_${g_version_1}.csv";
            if (-e "$TMP_DIR/kill.csv") {
                copy("$TMP_DIR/kill.csv", $kill_mutation_file) or die "La copia del archivo $kill_mutation_file falló: $!";
            }

            my $testMap_mutation_file = "$mutations_files_folder_by_version/testMap_${g_project_1}_${g_version_1}.csv";
            if (-e "$TMP_DIR/testMap.csv") {
                copy("$TMP_DIR/testMap.csv", $testMap_mutation_file) or die "La copia del archivo $testMap_mutation_file falló: $!";
            }

            # die "Finish mutation execution";
        }
    } 


    if ($should_execute_bug_detection) {
        my $buggy_version = "${g_version_num_1}b";
        my $project_with_buggy_version = "${g_project_1}_${buggy_version}";

        print "\n\n\nBUG DETECTION for $project_with_buggy_version:\n\n";

        $project->checkout_vid($buggy_version);
        $project->compile() or die "(Buggy version) Cannot compile sources for  buggy project_version $project_with_buggy_version";
        print "\n\n";

        my $path_bench_bug_detection  = "$project->{prog_root}/bench_bug_detection";
        if (!-e "$path_bench_bug_detection") {
            system("mkdir -p $path_bench_bug_detection");
        }

        copy_directory("$tmp_correct_tests_dir_by_version", $path_bench_bug_detection);

        # printf ("\n\n correct_tests_by_focal_class\n\n");
        # print Dumper(\%correct_tests_by_focal_class);
        # print "\n\n";

        while (my ($folder_correct_tests, $info_by_focal_class) = each %correct_tests_by_focal_class) {
            # Limpiamos los resultados de ejecuciones previas

            my $tests_dir_by_focal_class = "$path_bench_bug_detection/$folder_correct_tests";
            my $tests_dir_by_focal_class_escaped = quotemeta($tests_dir_by_focal_class);

            my $focal_class_buggy = $info_by_focal_class->{focal_class};
            my $total_correct_tests = $info_by_focal_class->{total_correct_tests};

            $num_buggy_version_executed_tests += $total_correct_tests;

            Utils::clean_test_results($project->{prog_root});

            printf ("$sep\n$focal_class_buggy  ($total_correct_tests correct tests)\n$sep\n");

            my $bug_results = {
                name => $project_with_buggy_version,
                project => $g_project_1,
                version => $buggy_version,
                version_num => $g_version_num_1,
                focal_class => $focal_class_buggy,
                total_executed_tests => $total_correct_tests,
                fail_compiling => "N/A",
                fail_executing => "N/A",
                has_failing_tests => "N/A",
                total_failing_tests => 0
            };


            my $compilation_log_buggy = Log::create_log("$TMP_DIR/comp_tests.log", ">")->{file_name};
            if (! $project->compile_ext_tests($tests_dir_by_focal_class_escaped, $compilation_log_buggy)) {
                printf ("\nBug detectado al compilar tests para la clase focal $focal_class_buggy \n\n\n");

                $bug_results->{fail_compiling} = "Yes";
                $num_buggy_version_fail_compiling += 1;
                push(@list_bug_detection, $bug_results);
                next;
            }
            $bug_results->{fail_compiling} = "No";

            my $log_run_tests_buggy = Log::create_log("$TMP_DIR/run_tests.log", ">")->{file_name};
            if (! $project->run_ext_tests($tests_dir_by_focal_class_escaped, "$INCL", $log_run_tests_buggy)) {
                printf ("\nBug detectado al ejecutar tests para la clase focal $focal_class_buggy \n\n\n");
                
                $bug_results->{fail_executing} = "Yes";
                $num_buggy_version_fail_executing += 1;
                push(@list_bug_detection, $bug_results);
                next;
            }
            $bug_results->{fail_executing} = "No";

            my $list_failings_buggy = Utils::get_failing_tests($log_run_tests_buggy) or die;
            my $count_buggy = scalar(@{$list_failings_buggy->{methods}}) + scalar(@{$list_failings_buggy->{classes}});

            if ($count_buggy > 0) {
                printf ("\nBug detectado en los asserts para la clase focal $focal_class_buggy \n\n\n");

                $bug_results->{has_failing_tests} = "Yes";
                $bug_results->{total_failing_tests} = $count_buggy; # por lo general es 1, pero puede darse que el modelo haya generado más de un test case
                $num_buggy_version_failing_tests += 1;
                push(@list_bug_detection, $bug_results);
                next;
            }
            $bug_results->{has_failing_tests} = "No";
            push(@list_bug_detection, $bug_results);

            printf ("\nNo se detectaron bugs para la clase focal $focal_class_buggy \n\n\n");
        }

        # Configuramos JSON::PP para mantener el orden de las claves y formatear la salida
        my $json = JSON::PP->new->pretty->canonical;
        # Convertimos la estructura de datos a JSON
        my $json_text = $json->encode(\%correct_tests_by_focal_class);
        # Guardamos el JSON en un archivo
        my $filename_correct_tests_json = "$PATH_CORRECT_TESTS/${g_project_1}_${g_version_1}_metadata_correct_tests.json";
        open(my $fh_json, '>', $filename_correct_tests_json) or die "No se pudo abrir el archivo '$filename_correct_tests_json' $!";
        print $fh_json $json_text;
        close($fh_json);

        # die "Finish bug detection";
    }
}



# Configuramos JSON::PP para mantener el orden de las claves y formatear la salida
my $json_focal_methods = JSON::PP->new->pretty->canonical;
# Convertimos la estructura de datos a JSON
my $json_focal_methods_text = $json_focal_methods->encode(\%focal_methods_by_focal_class_and_version);
# Guardamos el JSON en un archivo
my $filename_focal_methods_json = "$PATH_COVERAGE_FILES/${PID}_focal_methods_by_focal_class_and_version.json";
open(my $fh_json, '>', $filename_focal_methods_json) or die "No se pudo abrir el archivo '$filename_focal_methods_json' $!";
print $fh_json $json_focal_methods_text;
close($fh_json);



# Log current time
$SUMMARY_LOG->log_msg("");
$SUMMARY_LOG->log_time("End benchmark execution");
$SUMMARY_LOG->close();

$COMPILE_LOG->close();
$RUN_LOG->close();
$MUTATION_LOG->close();



# INIT SAVE LIST_RESULTS IN CSV FILE
# Ordenamos la lista de resultados
@list_results = sort {
    $a->{version_num} <=> $b->{version_num}
    || 
    $a->{test_id} <=> $b->{test_id}
} @list_results;

# Creamos una instancia de Text::CSV
my $csv_results = Text::CSV->new({ binary => 1, eol => "\n" });

open my $file_results, ">", "$PATH_LOGS/${PID}_01_report_execution.csv" or die "Cannot write methods hashes in CSV";
# Especificamos el orden de las columnas
my @sort_columns_report_execution = (
    'name',
    'project',
    'version',
    'version_num',
    'test_id',
    'focal_class',
    'focal_method',
    'total_suites_analyzed',
    'first_suite_id_correct_sintax',
    'first_suite_id_compilable',
    'first_suite_id_executable',
    'first_suite_id_passing_test',
    'first_suite_id_correct_test',
    'first_suite_id_focal_method_covered',
    'correct_syntax',
    'compilable',
    'executable',
    'passing_test',
    'correct_test',
    'focal_method_covered'
);


# Escribimos los nombres de las columnas en la primera línea
$csv_results->print($file_results, \@sort_columns_report_execution);
foreach my $hash (@list_results) {
    # Escribimos cada valor en el archivo
    $csv_results->print($file_results, [@{$hash}{@sort_columns_report_execution}]);
}
close $file_results;
# END SAVE LIST_RESULTS IN CSV FILE



# INIT SAVE LIST_COVERAGE_BY_TEST_ID IN CSV FILE
# Ordenamos la lista de resultados de coverage
@list_coverage_by_test_id = sort {
    $a->{version_num} <=> $b->{version_num}
    ||
    $a->{test_id} <=> $b->{test_id}
} @list_coverage_by_test_id;

open(CSV, ">$PATH_LOGS/${PID}_02_report_coverage_by_test_id.csv") or die "Cannot write output csv file $!";
print(CSV "Name,Project,Version,TestId,InstrumentClass,FocalMethod," .
            "LinesTotal,LinesCovered,RateLineCoverage," .
            "BranchesTotal,BranchesCovered,RateBranchCoverage," .
            "FocalMethodCovered\n");
foreach my $cov_test_id (@list_coverage_by_test_id) {
    printf(CSV "%s,%s,%s,%d,%s,%s,%d,%d,%s,%d,%d,%s,%s\n",
            $cov_test_id->{name},
            $cov_test_id->{project},
            $cov_test_id->{version},
            $cov_test_id->{test_id},
            $cov_test_id->{instrument_class},
            $cov_test_id->{focal_method},
            $cov_test_id->{lines_total},
            $cov_test_id->{lines_covered},
            #$cov_test_id->{line_coverage}, -> %f
            sprintf("%.2f", $cov_test_id->{line_coverage}),
            $cov_test_id->{branches_total},
            $cov_test_id->{branches_covered},
            #$cov_test_id->{condition_coverage} -> %f
            sprintf("%.2f", $cov_test_id->{branch_coverage}),
            $cov_test_id->{focal_method_covered}
    );
}
close(CSV);
# END SAVE LIST_COVERAGE_BY_TEST_ID IN CSV FILE



# INIT SAVE LIST_COVERAGE_BY_FOCAL_CLASS IN CSV FILE
# Ordenamos la lista de resultados de coverage
@list_coverage_by_focal_class = sort {
    $a->{version_num} <=> $b->{version_num}
    ||
    $a->{instrument_class} cmp $b->{instrument_class}
} @list_coverage_by_focal_class;

open(CSV, ">$PATH_LOGS/${PID}_03_report_coverage_by_focal_class.csv") or die "Cannot write output csv file $!";
print(CSV "Name,Project,Version,InstrumentClass," .
            "LinesTotal,LinesCovered,RateLineCoverage," .
            "BranchesTotal,BranchesCovered,RateBranchCoverage," .
            "MethodsTotal,MethodsCovered,RateMethodCoverage\n");
foreach my $cov_focal_class (@list_coverage_by_focal_class) {
    printf(CSV "%s,%s,%s,%s,%d,%d,%s,%d,%d,%s,%d,%d,%s\n",
            $cov_focal_class->{name},
            $cov_focal_class->{project},
            $cov_focal_class->{version},
            $cov_focal_class->{instrument_class},
            $cov_focal_class->{lines_total},
            $cov_focal_class->{lines_covered},
            sprintf("%.2f", $cov_focal_class->{line_coverage}),
            $cov_focal_class->{branches_total},
            $cov_focal_class->{branches_covered},
            sprintf("%.2f", $cov_focal_class->{branch_coverage}),
            $cov_focal_class->{methods_total},
            $cov_focal_class->{methods_covered},
            sprintf("%.2f", $cov_focal_class->{method_coverage}),
    );
}
close(CSV);
# END SAVE LIST_COVERAGE_BY_FOCAL_CLASS IN CSV FILE



# INIT SAVE LIST_MUTANTS IN CSV FILE
# Ordenamos la lista de resultados de mutantes
@list_mutants = sort {
    $a->{version_num} <=> $b->{version_num}
} @list_mutants;

open(CSV, ">$PATH_LOGS/${PID}_04_report_mutation.csv") or die "Cannot write output csv file $!";
print(CSV "Name,Project,Version,MutantsGenerated,MutantsCovered,MutantsKilled,MutationScoreKillOverCov,MutationScoreKillOverGen,MutationRunTime\n");
foreach my $mut (@list_mutants) {
    printf(CSV "%s,%s,%s,%d,%d,%d,%s,%s,%s\n",
            $mut->{name},
            $mut->{project},
            $mut->{version},
            $mut->{mutants_generated},
            $mut->{mutants_covered},
            $mut->{mutants_killed},
            sprintf("%.2f", $mut->{mutation_score_kill_over_cov}),
            sprintf("%.2f", $mut->{mutation_score_kill_over_gen}),
            sprintf("%.2f", $mut->{mutation_run_time})
    );
}
close(CSV);
# END SAVE LIST_MUTANTS IN CSV FILE



# INIT SAVE LIST_BUG_DETECTION IN CSV FILE
# Ordenamos la lista de resultados
@list_bug_detection = sort {
    $a->{version_num} <=> $b->{version_num}
    ||
    $a->{focal_class} cmp $b->{focal_class}
} @list_bug_detection;

# Creamos una instancia de Text::CSV
my $csv_bug_detection = Text::CSV->new({ binary => 1, eol => "\n" });

open my $file_bug_detection, ">", "$PATH_LOGS/${PID}_05_report_bug_detection.csv" or die "Cannot write methods hashes in CSV";
# Especificamos el orden de las columnas
my @sort_columns_report_bug_detection = (
    'name',
    'project',
    'version',
    'version_num',
    'focal_class',
    'total_executed_tests',
    'fail_compiling',
    'fail_executing',
    'has_failing_tests',
    'total_failing_tests'
);

# Escribimos los nombres de las columnas en la primera línea
$csv_bug_detection->print($file_bug_detection, \@sort_columns_report_bug_detection);
foreach my $hash (@list_bug_detection) {
    # Escribimos cada valor en el archivo
    $csv_bug_detection->print($file_bug_detection, [@{$hash}{@sort_columns_report_bug_detection}]);
}
close $file_bug_detection;
# END SAVE LIST_BUG_DETECTION IN CSV FILE



# INIT SAVE ANALYSIS COVERAGE SUMMARY BY PROJECT AND SAVE IN CSV FILE
# This generate a summary coverage report by Project
my $res_coverage_analysis = _run_analyze_coverage_by_d4j_project("Project", undef, undef, undef);
$COVERAGE_ANALYSIS_LOG->close();

my $total_lines = 0;
my $total_lines_covered = 0;
my $ratio_lines_covered = 0.0;
my $total_lines_covered_2 = 0;
my $ratio_lines_covered_2 = 0.0;

my $total_branches = 0;
my $total_branches_covered = 0;
my $ratio_branches_covered = 0.0;
my $total_branches_covered_2 = 0;
my $ratio_branches_covered_2 = 0.0;

my $total_methods = 0;
my $total_methods_covered = 0;
my $ratio_methods_covered = 0.0;
my $total_methods_covered_2 = 0;
my $ratio_methods_covered_2 = 0.0;

if ($res_coverage_analysis ne "Error") {
    (
        $coverage_project,
        $total_lines,
        $total_lines_covered,
        $ratio_lines_covered,
        $total_lines_covered_2,
        $ratio_lines_covered_2,
        $total_branches,
        $total_branches_covered,
        $ratio_branches_covered,
        $total_branches_covered_2,
        $ratio_branches_covered_2,
        $total_methods,
        $total_methods_covered,
        $ratio_methods_covered,
        $total_methods_covered_2,
        $ratio_methods_covered_2
    ) = split(/\|/, $res_coverage_analysis);
    $coverage_project = $PID;
} 
# else {
#     $coverage_project = "${PID}_Error_Analyzing_Coverage_Files";
# }

if ($res_coverage_analysis eq "Error" || $num_lines >= $total_lines) {
    $total_lines = $num_lines;
    $total_lines_covered = $num_lines_covered;
    $ratio_lines_covered = $total_lines == 0 ? 0.0 : $total_lines_covered/$total_lines*100;
    $total_branches = $num_branches;
    $total_branches_covered = $num_branches_covered;
    $ratio_branches_covered = $total_branches == 0 ? 0.0 : $total_branches_covered/$total_branches*100;
    $total_methods = $num_methods;
    $total_methods_covered = $num_methods_covered;
    $ratio_methods_covered = $total_methods == 0 ? 0.0 : $total_methods_covered/$total_methods*100;
}
# END SAVE ANALYSIS COVERAGE SUMMARY BY PROJECT AND SAVE IN CSV FILE



# INIT SHOW SUMMARY RESULTS IN LOG

print(STDERR "\n\n\n\n");
print(STDERR "SUMMARY RESULTS PROJECT: $PID\n\n");
print(STDERR "Total Focal Methods to test: $total_tests\n");
print(STDERR "Total test suites: $total_test_suites_to_analyze\n");
print(STDERR "\n\n\n");


my $rate_syntax_error = ($num_syntax_error * 100) / $total_tests;
my $rate_compile_error = ($num_compile_error * 100) / $total_tests;
my $rate_execute_error = ($num_execute_error * 100) / $total_tests;
my $rate_failing = ($num_failing * 100) / $total_tests;

print(STDERR "Tests with Syntax error: $num_syntax_error\n");
print(STDERR "% Tests with Syntax error: " . sprintf("%.2f", $rate_syntax_error)  . " %\n");
print(STDERR "Tests with Compile error: $num_compile_error\n");
print(STDERR "% Tests with Compile error: " . sprintf("%.2f", $rate_compile_error)  . " %\n");
print(STDERR "Tests with Execution error: $num_execute_error\n");
print(STDERR "% Tests with Execution error: " . sprintf("%.2f", $rate_execute_error)  . " %\n");
print(STDERR "Failing tests by class (Aux metric): $num_failing_by_class\n");
print(STDERR "Failing tests by method (Aux metric): $num_failing_by_method\n");
print(STDERR "Failing tests by class and method (Aux metric): $num_failing_by_class_and_method\n");
print(STDERR "Failing tests: $num_failing\n");
print(STDERR "% Failing tests: " . sprintf("%.2f", $rate_failing)  . " %\n\n");

my $rate_syntax_correct = ($num_syntax_correct * 100) / $total_tests;
my $rate_compile_correct = ($num_compile_correct * 100) / $total_tests;
my $rate_execute_correct = ($num_execute_correct * 100) / $total_tests;
my $rate_passing = ($num_passing * 100) / $total_tests;
my $rate_correct = ($num_correct * 100) / $total_tests;
my $rate_focal_method_covered = ($num_focal_method_covered * 100) / $total_tests;

print(STDERR "Num Correct syntax: $num_syntax_correct\n");
print(STDERR "% Correct syntax: " . sprintf("%.2f", $rate_syntax_correct)  . " %\n");
print(STDERR "Num Compilable tests: $num_compile_correct\n");
print(STDERR "% Compilable tests: " . sprintf("%.2f", $rate_compile_correct)  . " %\n");
print(STDERR "Num Executable tests: $num_execute_correct\n");
print(STDERR "% Executable tests: " . sprintf("%.2f", $rate_execute_correct)  . " %\n");
print(STDERR "Num Passing tests: $num_passing\n");
print(STDERR "% Passing tests: " . sprintf("%.2f", $rate_passing)  . " %\n");
print(STDERR "Num Correct tests: $num_correct\n");
print(STDERR "% Correct tests: " . sprintf("%.2f", $rate_correct)  . " %\n");
print(STDERR "Num Focal methods covered: $num_focal_method_covered\n");
print(STDERR "% Focal methods covered: " . sprintf("%.2f", $rate_focal_method_covered)  . " %\n");
print(STDERR "\n\n\n");


print(STDERR "Coverage results:\n\n");
print(STDERR "Total lines: $total_lines\n");
print(STDERR "Total lines covered (Strategy 1): $total_lines_covered\n");
print(STDERR "% Line coverage (Strategy 1):     " . sprintf("%.2f", $ratio_lines_covered)  . " %\n");
print(STDERR "Total lines covered (Strategy 2): $total_lines_covered_2\n");
print(STDERR "% Line coverage (Strategy 2):     " . sprintf("%.2f", $ratio_lines_covered_2)  . " %\n\n");
print(STDERR "Total branches: $total_branches\n");
print(STDERR "Total branches covered (Strategy 1): $total_branches_covered\n");
print(STDERR "% Branch coverage (Strategy 1):      " . sprintf("%.2f", $ratio_branches_covered)  . " %\n");
print(STDERR "Total branches covered (Strategy 2): $total_branches_covered_2\n");
print(STDERR "% Branch coverage (Strategy 2):      " . sprintf("%.2f", $ratio_branches_covered_2)  . " %\n\n");
print(STDERR "Total methods: $total_methods\n");
print(STDERR "Total methods covered (Strategy 1): $total_methods_covered\n");
print(STDERR "% Method coverage (Strategy 1):     " . sprintf("%.2f", $ratio_methods_covered)  . " %\n");
print(STDERR "Total methods covered (Strategy 2): $total_methods_covered_2\n");
print(STDERR "% Method coverage (Strategy 2):     " . sprintf("%.2f", $ratio_methods_covered_2)  . " %\n");
print(STDERR "\n\n\n");


my $rate_kill_over_gen = $num_mutants_generated == 0 ? 0.0 : $num_mutants_killed/$num_mutants_generated*100;
my $rate_kill_over_cov = $num_mutants_covered == 0 ? 0.0 : $num_mutants_killed/$num_mutants_covered*100;

print(STDERR "Mutation results:\n\n");
print(STDERR "Num Mutants Generated: $num_mutants_generated\n");
print(STDERR "Num Mutants Covered: $num_mutants_covered\n");
print(STDERR "Num Mutants Killed: $num_mutants_killed\n");
print(STDERR "% Kill over covered mutants: " . sprintf("%.2f", $rate_kill_over_cov)  . " %\n");
print(STDERR "% Kill over generated mutants: " . sprintf("%.2f", $rate_kill_over_gen)  . " %\n");
print(STDERR "\n\n\n");


$num_buggy_version_bugs_detected = $num_buggy_version_fail_compiling + $num_buggy_version_fail_executing + $num_buggy_version_failing_tests;
my $rate_buggy_version_fail_compiling = $num_buggy_version_executed_tests == 0 ? 0.0 : $num_buggy_version_fail_compiling/$num_buggy_version_executed_tests*100;
my $rate_buggy_version_fail_executing = $num_buggy_version_executed_tests == 0 ? 0.0 : $num_buggy_version_fail_executing/$num_buggy_version_executed_tests*100;
my $rate_buggy_version_failing_tests = $num_buggy_version_executed_tests == 0 ? 0.0 : $num_buggy_version_failing_tests/$num_buggy_version_executed_tests*100;
my $rate_buggy_version_bugs_detected = $num_buggy_version_executed_tests == 0 ? 0.0 : $num_buggy_version_bugs_detected/$num_buggy_version_executed_tests*100;

print(STDERR "Bug detection results:\n\n");
print(STDERR "Total executed tests on buggy version: $num_buggy_version_executed_tests\n");
print(STDERR "Num fail compiling on buggy version: $num_buggy_version_fail_compiling\n");
print(STDERR "% fail compiling on buggy version: " . sprintf("%.2f", $rate_buggy_version_fail_compiling)  . " %\n");
print(STDERR "Num fail executing on buggy version: $num_buggy_version_fail_executing\n");
print(STDERR "% fail executing on buggy version: " . sprintf("%.2f", $rate_buggy_version_fail_executing)  . " %\n");
print(STDERR "Num failing tests on buggy version:  $num_buggy_version_failing_tests\n");
print(STDERR "% failing tests on buggy version: " . sprintf("%.2f", $rate_buggy_version_failing_tests)  . " %\n");
print(STDERR "Total bugs detected on buggy version: $num_buggy_version_bugs_detected\n");
print(STDERR "% bugs detected on buggy version: " . sprintf("%.2f", $rate_buggy_version_bugs_detected)  . " %\n");
print(STDERR "\n\n");

# END SHOW SUMMARY RESULTS IN LOG



# INIT SAVE EXECUTION RESULTS IN RESULTS_LOG
$RESULTS_LOG->log_msg("EXECUTION RESULTS PROJECT: $PID");
$RESULTS_LOG->log_msg("");
$RESULTS_LOG->log_msg("- Total Focal Methods to test: $total_tests");
$RESULTS_LOG->log_msg("- Total test suites: $total_test_suites_to_analyze");
$RESULTS_LOG->log_msg("");
$RESULTS_LOG->log_msg("");
$RESULTS_LOG->log_msg("");
$RESULTS_LOG->log_msg("- Tests with Syntax error: $num_syntax_error");
$RESULTS_LOG->log_msg("- Rate Tests with Syntax error: " . sprintf("%.2f", $rate_syntax_error)  . " %");
$RESULTS_LOG->log_msg("- Tests with Compile error: $num_compile_error");
$RESULTS_LOG->log_msg("- Rate Tests with Compile error: " . sprintf("%.2f", $rate_compile_error)  . " %");
$RESULTS_LOG->log_msg("- Tests with Execution error: $num_execute_error");
$RESULTS_LOG->log_msg("- Rate Tests with Execution error: " . sprintf("%.2f", $rate_execute_error)  . " %");
$RESULTS_LOG->log_msg("- Failing tests by class (Aux metric): $num_failing_by_class");
$RESULTS_LOG->log_msg("- Failing tests by method (Aux metric): $num_failing_by_method");
$RESULTS_LOG->log_msg("- Failing tests by class and method (Aux metric): $num_failing_by_class_and_method");
$RESULTS_LOG->log_msg("- Failing tests: $num_failing");
$RESULTS_LOG->log_msg("- Rate Failing tests: " . sprintf("%.2f", $rate_failing)  . " %");
$RESULTS_LOG->log_msg("");
$RESULTS_LOG->log_msg("- Correct syntax: $num_syntax_correct");
$RESULTS_LOG->log_msg("- Rate Correct syntax: " . sprintf("%.2f", $rate_syntax_correct)  . " %");
$RESULTS_LOG->log_msg("- Compilable tests: $num_compile_correct");
$RESULTS_LOG->log_msg("- Rate Compilable tests: " . sprintf("%.2f", $rate_compile_correct)  . " %");
$RESULTS_LOG->log_msg("- Executable tests: $num_execute_correct");
$RESULTS_LOG->log_msg("- Rate Executable tests: " . sprintf("%.2f", $rate_execute_correct)  . " %");
$RESULTS_LOG->log_msg("- Passing tests: $num_passing");
$RESULTS_LOG->log_msg("- Rate Passing tests: " . sprintf("%.2f", $rate_passing)  . " %");
$RESULTS_LOG->log_msg("- Correct tests: $num_correct");
$RESULTS_LOG->log_msg("- Rate Correct tests: " . sprintf("%.2f", $rate_correct)  . " %");
$RESULTS_LOG->log_msg("- Focal methods covered: $num_focal_method_covered");
$RESULTS_LOG->log_msg("- Rate Focal methods covered: " . sprintf("%.2f", $rate_focal_method_covered)  . " %");
$RESULTS_LOG->log_msg("");
$RESULTS_LOG->log_msg("");
$RESULTS_LOG->log_msg("");
$RESULTS_LOG->log_msg("Coverage Results");
$RESULTS_LOG->log_msg("");
$RESULTS_LOG->log_msg("- Total lines: $total_lines");
$RESULTS_LOG->log_msg("- Total lines covered (Strategy 1): $total_lines_covered");
$RESULTS_LOG->log_msg("- Rate Line coverage (Strategy 1):  " . sprintf("%.2f", $ratio_lines_covered)  . " %");
$RESULTS_LOG->log_msg("- Total lines covered (Strategy 2): $total_lines_covered_2");
$RESULTS_LOG->log_msg("- Rate Line coverage (Strategy 2):  " . sprintf("%.2f", $ratio_lines_covered_2)  . " %");
$RESULTS_LOG->log_msg("");
$RESULTS_LOG->log_msg("- Total branches: $total_branches");
$RESULTS_LOG->log_msg("- Total branches covered (Strategy 1): $total_branches_covered");
$RESULTS_LOG->log_msg("- Rate Branch coverage (Strategy 1):   " . sprintf("%.2f", $ratio_branches_covered)  . " %");
$RESULTS_LOG->log_msg("- Total branches covered (Strategy 2): $total_branches_covered_2");
$RESULTS_LOG->log_msg("- Rate Branch coverage (Strategy 2):   " . sprintf("%.2f", $ratio_branches_covered_2)  . " %");
$RESULTS_LOG->log_msg("");
$RESULTS_LOG->log_msg("- Total methods: $total_methods");
$RESULTS_LOG->log_msg("- Total methods covered (Strategy 1): $total_methods_covered");
$RESULTS_LOG->log_msg("- Rate Method coverage (Strategy 1):  " . sprintf("%.2f", $ratio_methods_covered)  . " %");
$RESULTS_LOG->log_msg("- Total methods covered (Strategy 2): $total_methods_covered_2");
$RESULTS_LOG->log_msg("- Rate Method coverage (Strategy 2):  " . sprintf("%.2f", $ratio_methods_covered_2)  . " %");
$RESULTS_LOG->log_msg("");
$RESULTS_LOG->log_msg("");
$RESULTS_LOG->log_msg("");
$RESULTS_LOG->log_msg("Mutation Results");
$RESULTS_LOG->log_msg("");
$RESULTS_LOG->log_msg("- Mutants Generated: $num_mutants_generated");
$RESULTS_LOG->log_msg("- Mutants Covered: $num_mutants_covered");
$RESULTS_LOG->log_msg("- Mutants Killed: $num_mutants_killed");
$RESULTS_LOG->log_msg("- Rate Kill over covered mutants: " . sprintf("%.2f", $rate_kill_over_cov)  . " %");
$RESULTS_LOG->log_msg("- Rate Kill over generated mutants: " . sprintf("%.2f", $rate_kill_over_gen)  . " %");
$RESULTS_LOG->log_msg("");
$RESULTS_LOG->log_msg("");
$RESULTS_LOG->log_msg("");
$RESULTS_LOG->log_msg("Bug Detection Results");
$RESULTS_LOG->log_msg("");
$RESULTS_LOG->log_msg("- Total executed tests on buggy version: $num_buggy_version_executed_tests");
$RESULTS_LOG->log_msg("- Num fail compiling on buggy version: $num_buggy_version_fail_compiling");
$RESULTS_LOG->log_msg("- Rate fail compiling on buggy version: " . sprintf("%.2f", $rate_buggy_version_fail_compiling)  . " %");
$RESULTS_LOG->log_msg("- Num fail executing on buggy version: $num_buggy_version_fail_executing");
$RESULTS_LOG->log_msg("- Rate fail executing on buggy version: " . sprintf("%.2f", $rate_buggy_version_fail_executing)  . " %");
$RESULTS_LOG->log_msg("- Num failing tests on buggy version: $num_buggy_version_failing_tests");
$RESULTS_LOG->log_msg("- Rate failing tests on buggy version: " . sprintf("%.2f", $rate_buggy_version_failing_tests)  . " %");
$RESULTS_LOG->log_msg("- Total bugs detected on buggy version: $num_buggy_version_bugs_detected");
$RESULTS_LOG->log_msg("- Rate bugs detected on buggy version: " . sprintf("%.2f", $rate_buggy_version_bugs_detected)  . " %");

# $RESULTS_LOG->close();
# END SAVE EXECUTION RESULTS IN RESULTS_LOG



# INIT SAVE SUMMARY RESULT BY PROJECT IN CSV FILE
# Imprimimos el summary de los resultados
my $file_all_summary_results = "$PATH_LOGS/01_all_summary_results.csv";
my $already_exists_file_summary_res = 0;
if (-e "$file_all_summary_results") {
    $already_exists_file_summary_res = 1;
}

# Con '>' se sobreescribe el archivo, con '>>' se agregan registros al archivo existente
open(CSV, ">>$file_all_summary_results") or die "Cannot write output $file_all_summary_results csv file $!";
if (! $already_exists_file_summary_res) {
    # Imprimimos el head solo si el archivo no existía antes
    print(CSV "project,total_tests,syntax_error,rate_syntax_error,compile_error,rate_compile_error,execute_error,rate_execute_error,failing_tests,rate_failing_tests," . 
                "syntax_correct,rate_syntax_correct,compile_correct,rate_compile_correct,passing_tests,rate_passing_tests,correct_tests,rate_correct_tests" .
                "focal_methods_covered,rate_focal_methods_covered\n");
}
printf(CSV "%s,%d,%d,%s,%d,%s,%d,%s,%d,%s,%d,%s,%d,%s,%d,%s,%d,%s,%d,%s\n",
        $PID,
        $total_tests,
        $num_syntax_error,
        sprintf("%.2f", $rate_syntax_error),
        $num_compile_error,
        sprintf("%.2f", $rate_compile_error),
        $num_execute_error,
        sprintf("%.2f", $rate_execute_error),
        $num_failing,
        sprintf("%.2f", $rate_failing),
        $num_syntax_correct,
        sprintf("%.2f", $rate_syntax_correct),
        $num_compile_correct,
        sprintf("%.2f", $rate_compile_correct),
        $num_passing,
        sprintf("%.2f", $rate_passing),
        $num_correct,
        sprintf("%.2f", $rate_correct),
        $num_focal_method_covered,
        sprintf("%.2f", $rate_focal_method_covered)
);
close(CSV);
# END SAVE SUMMARY RESULT BY PROJECT IN CSV FILE



# INIT SAVE SUMMARY COVERAGE BY PROJECT IN CSV FILE
my $file_all_summary_coverage = "$PATH_LOGS/02_all_summary_coverage.csv";
my $already_exists_file_summary_cov = 0;
if (-e "$file_all_summary_coverage") {
    $already_exists_file_summary_cov = 1;
}

open(CSV, ">>$file_all_summary_coverage") or die "Cannot write output $file_all_summary_coverage csv file $!";
if (! $already_exists_file_summary_cov) {
    # Imprimimos el head solo si el archivo no existía antes
    print(CSV "project,total_lines,total_lines_covered,ratio_lines_covered,total_lines_covered_2,ratio_lines_covered_2," . 
                "total_branches,total_branches_covered,ratio_branches_covered,total_branches_covered_2,ratio_branches_covered_2," .
                "total_methods,total_methods_covered,ratio_methods_covered,total_methods_covered_2,ratio_methods_covered_2\n");
}
printf(CSV "%s,%d,%d,%s,%d,%s,%d,%d,%s,%d,%s,%d,%d,%s,%d,%s\n",
        $PID,
        $total_lines,
        $total_lines_covered,
        sprintf("%.2f", $ratio_lines_covered),
        $total_lines_covered_2,
        sprintf("%.2f", $ratio_lines_covered_2),
        $total_branches,
        $total_branches_covered,
        sprintf("%.2f", $ratio_branches_covered),
        $total_branches_covered_2,
        sprintf("%.2f", $ratio_branches_covered_2),
        $total_methods,
        $total_methods_covered,
        sprintf("%.2f", $ratio_methods_covered),
        $total_methods_covered_2,
        sprintf("%.2f", $ratio_methods_covered_2)
);
close(CSV);
# END SAVE SUMMARY COVERAGE BY PROJECT IN CSV FILE



# INIT SAVE SUMMARY MUTATION BY PROJECT IN CSV FILE
my $file_all_summary_mutation = "$PATH_LOGS/03_all_summary_mutation.csv";
my $already_exists_file_summary_mut = 0;
if (-e "$file_all_summary_mutation") {
    $already_exists_file_summary_mut = 1;
}

open(CSV, ">>$file_all_summary_mutation") or die "Cannot write output $file_all_summary_mutation csv file $!";
if (! $already_exists_file_summary_mut) {
    # Imprimimos el head solo si el archivo no existía antes
    print(CSV "project,mutants_generated,mutants_covered,mutants_killed,rate_kill_over_cov,rate_kill_over_gen\n");
}
printf(CSV "%s,%d,%d,%d,%s,%s\n",
        $PID,
        $num_mutants_generated,
        $num_mutants_covered,
        $num_mutants_killed,
        sprintf("%.2f", $rate_kill_over_cov),
        sprintf("%.2f", $rate_kill_over_gen)
);
close(CSV);
# INIT SAVE SUMMARY MUTATION BY PROJECT IN CSV FILE



# INIT SAVE SUMMARY BUG_DETECTION BY PROJECT IN CSV FILE
my $file_all_summary_bug_detection = "$PATH_LOGS/04_all_summary_bug_detection.csv";
my $already_exists_file_summary_bug = 0;
if (-e "$file_all_summary_bug_detection") {
    $already_exists_file_summary_bug = 1;
}

open(CSV, ">>$file_all_summary_bug_detection") or die "Cannot write output $file_all_summary_bug_detection csv file $!";
if (! $already_exists_file_summary_bug) {
    # Imprimimos el head solo si el archivo no existía antes
    print(CSV "project,total_executed_tests_on_buggy_version," . 
                "num_fail_compiling,rate_fail_compiling," . 
                "num_fail_executing,rate_fail_executing," . 
                "num_failing_tests,rate_failing_tests," . 
                "total_bugs_detected,rate_bugs_detected\n");
}
printf(CSV "%s,%d,%d,%s,%d,%s,%d,%s,%d,%s\n",
        $PID,
        $num_buggy_version_executed_tests,
        $num_buggy_version_fail_compiling,
        sprintf("%.2f", $rate_buggy_version_fail_compiling),
        $num_buggy_version_fail_executing,
        sprintf("%.2f", $rate_buggy_version_fail_executing),
        $num_buggy_version_failing_tests,
        sprintf("%.2f", $rate_buggy_version_failing_tests),
        $num_buggy_version_bugs_detected,
        sprintf("%.2f", $rate_buggy_version_bugs_detected),
);
close(CSV);
# INIT SAVE SUMMARY BUG_DETECTION BY PROJECT IN CSV FILE



# Clean up
system("rm -rf $PATH_INSTRUMENTS") unless $DEBUG;
system("rm -rf $PATH_CORRECT_TESTS_TMP") unless $DEBUG;
system("rm -rf $TMP_DIR") unless $DEBUG;

# Calcula el tiempo transcurrido
my $elapsed = tv_interval($start_time);
# Convierte el tiempo a horas, minutos y segundos
my $hours   = int($elapsed / 3600);
my $minutes = int(($elapsed % 3600) / 60);
my $seconds = int($elapsed % 60);

my $formatted_elapsed = sprintf("%02d:%02d:%02d", $hours, $minutes, $seconds);

# printf("\nTiempo de ejecucion project ${PID}: %02d:%02d:%02d\n\n", $hours, $minutes, $seconds);
print "\nTotal execution time project ${PID}: $formatted_elapsed\n\n\n";

$RESULTS_LOG->log_msg("");
$RESULTS_LOG->log_msg("");
$RESULTS_LOG->log_msg("");
$RESULTS_LOG->log_msg("=> Total execution time: $formatted_elapsed");
$RESULTS_LOG->log_msg("");
$RESULTS_LOG->close();









#
# Remove uncompilable test cases based on the compiler's log (if there
# is any issue non-related to any test case, the correspondent source
# file is removed)
#
sub _rm_classes {
    my ($comp_log, $src, $file_name, $test_method_name) = @_;

    open(LOG, "<$comp_log") or die "Cannot read compiler log!";
    $SUMMARY_LOG->log_msg(" - Catching uncompilable test method(s): $file_name");
    my $num_uncompilable_test_classes = 0;
    my @uncompilable_tests = ();
    my $error;

    while (<LOG>) {
        #my $removed = 0;
        my $find_method_test_error = 0;

        # Find file names in javac's log: [javac] "path"/"file_name".java:"line_number": error: "error_text"
        #next unless /javac.*($TMP_DIR\/$src\/(.*\.java)):(\d+):.*error/;
        #next unless /javac.*($TMP_DIR\/$src\/(.*\.java)):(\d+):.*(error:.([^-]+))/;
        # next unless /javac.*($TMP_DIR\/$src\/(.*\.java)):(\d+):.*(error:.([^\.]+))/;
        next unless /javac.*($TMP_DIR\/$src\/(.*\.java)):(\d+):.*(error:.([^\n]+))/;
        my $file = $1;
        my $class = $2;
        my $line_error_number = $3;
        # Error es todo lo que haya después de 'error: ' hasta que encuentre un salto de línea
        $error = $5;

        #print(STDERR "File before: $file\n");
        #print(STDERR "Class inmediatly: $class\n");

        # /javac.*(/tmp/checkouts/replace_generated_tests.pl_50685_1706071672\/generated-tests\/(.*\.java)):(\d+):.*error/

        # Skip already removed files
        next unless -e $file;

        $class =~ s/\.java$//; # Le quita la extensión .java
        #print(STDERR "Class after .java: $class\n");

        $class =~ s/\//\./g; # Reemplaza los '/' por '.'
        #print(STDERR "Class after puntos: $class\n");

        my $test_name_with_error = "";
        # if ($test_method_name eq "There Are No Methods" || $test_method_name eq "More Than One") {
        if ($test_method_name eq "There Are No Methods" || $test_method_name =~ /\|/) {
            # To which test method does the uncompilable line belong?
            open(JAVA_FILE, $file) or die "Cannot open '$file' file!";
            my $line_index = 0;
            while (<JAVA_FILE>) {
                ++$line_index;
                next unless /public\s*void\s*(test.*)\s*\(\s*\).*/;
                my $t_name = $1;

                if ($line_index > $line_error_number) {
                    # Si la línea que se está leyendo es mayor que la línea del error, finalizamos
                    last;
                }

                $test_name_with_error = $t_name;
                #$removed = 1;
                $find_method_test_error = 1;
            }
            close(JAVA_FILE);
        } else {
            $test_name_with_error = $test_method_name;
            #$removed = 1;
            $find_method_test_error = 1;
        }

        #print(STDERR "File after: $file\n");

        #if (! $removed) {
        if (! $find_method_test_error) {
            # in case of compilation issues due to, for example, wrong
            # or non-existing imported classes, or problems with any
            # super class, or the generated name test has wrong pattern,
            # the source file is removed
            $SUMMARY_LOG->log_msg("   $class");
            $SUMMARY_LOG->log_msg("   => Not suitable compilable test method in class caused by error: $error");

            #$file =~ s/\$/\\\$/g; # Escapamos el caracter especial $
            #system("mv $file $file.broken") == 0 or die "Cannot rename uncompilable source file";

            # get rid of all test cases of this class that have been
            # selected to be removed
            @uncompilable_tests = grep ! /^--- ${class}::/, @uncompilable_tests;
            # Update counter
            ++$num_uncompilable_test_classes;
        } else {
            # e.g., '--- org.foo.BarTest::test09'
            my $test_canonical_name = "     --- ${class}::${test_name_with_error}";
            # Skip already selected (to be removed) test cases
            if (! grep{/^$test_canonical_name$/} @uncompilable_tests) {
                push(@uncompilable_tests, $test_canonical_name);
            }
        }
    }
    close(LOG);

    if (scalar(@uncompilable_tests) > 0) {
        # Write to a file the name of all uncompilable test cases (one per
        # line) and call 'rm_broken_tests.pl' to remove all of them
        my $uncompilable_tests_file_path = "$TMP_DIR/uncompilable-test-cases.txt";
        open my $uncompilable_tests_file, ">$uncompilable_tests_file_path" or die $!;
        print $uncompilable_tests_file join("\n", @uncompilable_tests);
        close($uncompilable_tests_file);

        $SUMMARY_LOG->log_file("   - Finding " . scalar(@uncompilable_tests) . " uncompilable test method(s):", $uncompilable_tests_file_path);
        $SUMMARY_LOG->log_msg("     => Not compilable test method caused by error: $error");
        $SUMMARY_LOG->log_msg("");
        #$SUMMARY_LOG->log_msg("\n");
        
        # Con esta línea se intenta remover el assert que falla, pero se requiere que el test esté en varias líneas
        #Utils::exec_cmd("export D4J_RM_ASSERTS=$RM_ASSERTS && $UTIL_DIR/rm_broken_tests.pl $uncompilable_tests_file_path $TMP_DIR/$src", "Remove uncompilable test method(s)")
        #        or die "Cannot remove uncompilable test method(s)";
    }

    return (scalar(@uncompilable_tests), $num_uncompilable_test_classes);
}


sub _get_result_parser_utils {
    my ($test_java_file, $function) = @_;

    my $python_script = 'parser_utils.py';
    #my $src_code = 'example source code';
    # Construimos la línea de comando
    my @command = (
        'python3', $python_script,
        '--java_file', $test_java_file,
        #'--src_code', $src_code,
        '--function', $function
    );

    # Ejecutamos el script Python y capturamos la salida
    my $stdout = gensym;
    my $stderr = gensym;
    my $pid = open3(undef, $stdout, $stderr, @command);
    #my $pid = open3(undef, $stdout, $stderr, 'python3', $python_script, $src_code);

    # Leemos la salida estándar
    my $res_script_python = do { local $/; <$stdout> };
    chomp $res_script_python;

    # Leemos la salida de error (si hay)
    my $error_output = do { local $/; <$stderr> };
    chomp $error_output;

    # Esperamos a que el proceso termine
    waitpid($pid, 0);

    # Imprimimos el resultado
    #print "Resultado del script Python: $res_script_python\n";
    if ($error_output) {
        print "Errores del script parser_utils.py: $error_output\n";
    }

    return $res_script_python;
}


sub _validate_method_coverage {
    my ($focal_method_name, $path_xml_coverage_file) = @_;

    # python3 validate_method_coverage.py \
    # --focal_method_name "setLenient" \
    # --path_xml_coverage_file "/d4j_generated_tests_only_public_bs/prueba_results/coverage_pGson_v4f_t0.xml" 

    my $python_script = 'validate_method_coverage.py';
    #my $src_code = 'example source code';
    # Construimos la línea de comando
    my @command = (
        'python3', $python_script,
        '--focal_method_name', $focal_method_name,
        '--path_xml_coverage_file', $path_xml_coverage_file
    );

    # Ejecutamos el script Python y capturamos la salida
    my $stdout = gensym;
    my $stderr = gensym;
    my $pid = open3(undef, $stdout, $stderr, @command);
    #my $pid = open3(undef, $stdout, $stderr, 'python3', $python_script, $src_code);

    # Leemos la salida estándar
    my $res_script_python = do { local $/; <$stdout> };
    chomp $res_script_python;

    # Leemos la salida de error (si hay)
    my $error_output = do { local $/; <$stderr> };
    chomp $error_output;

    # Esperamos a que el proceso termine
    waitpid($pid, 0);

    # Imprimimos el resultado
    #print "Resultado del script Python: $res_script_python\n";
    if ($error_output) {
        print "Errores del script validate_method_coverage.py: $error_output\n";
    }

    return $res_script_python;
}


sub _run_analyze_coverage_by_d4j_project {
    my ($analysis_type, $test_id, $path_xml_coverage_file, $focal_methods_of_a_focal_class) = @_;
    # python3 analyze_coverage_by_d4j_project.py \
    #     --project "Gson" \
    #     --analysis_type "Project"
    #     --path_coverage_files "/d4j_generated_tests_only_public_bs/prueba_results/coverage_files" \
    #     --json_file_focal_methods "/d4j_generated_tests_only_public_bs/prueba_results/coverage_files/Gson_focal_methods_by_focal_class_and_version.json"
    #     --path_xml_coverage_file "/d4j_generated_tests_only_public_bs/prueba_results/coverage.xml"
    #     --focal_methods_of_a_focal_class "someMethod1|someMethod2|someMethod3"
    #     --output_path "/d4j_generated_tests_only_public_bs/prueba_results/coverage_files"

    my $python_script = 'analyze_coverage_by_d4j_project.py';
    #my $src_code = 'example source code';
    # Construimos la línea de comando
    my @command = (
        'python3', $python_script,
        '--project', $PID,
        '--analysis_type', $analysis_type,
        '--path_coverage_files', "$PATH_LOGS/coverage_files",
        '--json_file_focal_methods', "$PATH_COVERAGE_FILES/${PID}_focal_methods_by_focal_class_and_version.json",
        '--output_path', $PATH_LOGS
    );

    if (defined $test_id) {
        push(@command, '--test_id');
        push(@command, $test_id);
    }

    if (defined $path_xml_coverage_file) {
        push(@command, '--path_xml_coverage_file');
        push(@command, $path_xml_coverage_file);
    }

    if (defined $focal_methods_of_a_focal_class) {
        push(@command, '--focal_methods_of_a_focal_class');
        push(@command, $focal_methods_of_a_focal_class);
    }

    # Ejecutamos el script Python y capturamos la salida
    my $stdout = gensym;
    my $stderr = gensym;
    my $pid = open3(undef, $stdout, $stderr, @command);

    # Leemos la salida estándar
    my $res_script_python = do { local $/; <$stdout> };
    chomp $res_script_python;

    # Leemos la salida de error (si hay)
    my $error_output = do { local $/; <$stderr> };
    chomp $error_output;

    # Esperamos a que el proceso termine
    waitpid($pid, 0);

    # Imprimimos el resultado
    #print "Resultado del script Python: $res_script_python\n";
    if ($error_output) {
        print "Errores del script analyze_coverage_by_d4j_project.py: $error_output\n";
        $COVERAGE_ANALYSIS_LOG->log_msg("=> Failing execution analysis coverage for project: $PID\n\n\n");
        $COVERAGE_ANALYSIS_LOG->log_msg($error_output);
        $COVERAGE_ANALYSIS_LOG->log_msg("");

        return "Error";
    }

    return $res_script_python;
}



# Define una función de comparación personalizada para ordenar las claves
sub compare_keys_by_version {
    # La key contiene los siguientes valores: pid|vid|vid_num
    # Ordenamos solo por vid_num
    my ($a_vid_num) = (split('\|', $a))[2];
    my ($b_vid_num) = (split('\|', $b))[2];

    $a_vid_num <=> $b_vid_num;
}


# Define una función de comparación personalizada para ordenar las claves
sub compare_keys_by_version_and_test_id {
    # La key contiene los siguientes valores: pid|vid|vid_num|tid
    # Ordenamos solo por vid_num y tid
    my ($a_vid_num, $a_tid) = (split('\|', $a))[2, 3];
    my ($b_vid_num, $b_tid) = (split('\|', $b))[2, 3];

    $a_vid_num <=> $b_vid_num
        || $a_tid <=> $b_tid;
}


sub template_validation_property {
    my ($error_msg) = @_;
    if ($VALIDATE_TEMPLATE) {
        die "$error_msg";
    }
}

sub _run_coverage {

    # # Run Coverage only on Compiled tests
    # my $src_dir = $project->src_dir($vid);
    # my $SINGLE_TEST = $test_canonical_name;
    # # Clean temporary files that hold test results
    # my $fail_tests_log = "$TMP_DIR/$FILE_FAILING_TESTS";
    # Utils::clean_test_results($TMP_DIR);

    # my $cov_results = Coverage::coverage_ext($project, $file_instrument_classes, $src_dir, $project_test_dir, "*.java", $fail_tests_log, $SINGLE_TEST);
    # Utils::clean_test_results($TMP_DIR);

    # $cov_results->{line_coverage} = $cov_results->{lines_covered}/$cov_results->{lines_total}*100;
    # $cov_results->{condition_coverage} = ($cov_results->{branches_total} == 0 ? 0 : $cov_results->{branches_covered}/$cov_results->{branches_total}*100);

    # $cov_results->{name} = $file_name;
    # $cov_results->{project} = $pid;
    # $cov_results->{version} = $vid;
    # $cov_results->{test_id} = $tid;

    # push(@list_coverage_by_test_id, $cov_results);

    #unlink $file_instrument_classes or die "No se pudo eliminar el archivo $file_instrument_classes: $!\n";

    # printf("%18s: %d\n",     "Lines total",        $cov_results->{lines_total});
    # printf("%18s: %d\n",     "Lines covered",      $cov_results->{lines_covered});
    # printf("%18s: %d\n",     "Conditions total",   $cov_results->{branches_total});
    # printf("%18s: %d\n",     "Conditions covered", $cov_results->{branches_covered});
    # printf("%18s: %.1f%%\n", "Line coverage",      $cov_results->{lines_covered}/$cov_results->{lines_total}*100);
    # printf("%18s: %.1f%%\n", "Condition coverage", ($cov_results->{branches_total} == 0 ? 0 : $cov_results->{branches_covered}/$cov_results->{branches_total}*100));

}


#
# Parse coverage log file and return reference to a hash that holds all results
#
sub _get_info_from_xml {
    my ($xml, ) = @_;
    my ($lt, $lc, $bt, $bc);

    -e $xml or die "Result xml file does not exist: $xml!";

    # Parse XML file
    open FH, $xml;
    while (<FH>) {
        if (/lines-covered="(\d+)" lines-valid="(\d+)" branches-covered="(\d+)" branches-valid="(\d+)"/) {
            ($lc, $lt, $bc, $bt) = ($1, $2, $3, $4);
        }
    }
    close FH;

    die "values not set" unless defined $lt;

    # Set all values and return hash reference
    return {
        $LINES_TOTAL => $lt,
        $LINES_COVERED => $lc,
        $BRANCHES_TOTAL => $bt,
        $BRANCHES_COVERED => $bc,
    };
}



sub _clean_mutation_results {
    my ($work_dir) = @_;

    # Remove the files that list all mutants results
    my $excl_file = "$work_dir/$EXCL_FILE";
    my $kill_file = "$work_dir/$KILL_FILE";
    my $summary_file = "$work_dir/$SUMMARY_FILE";
    my $test_map_file = "$work_dir/$TEST_MAP_FILE";

    if (-e "$excl_file") {
        unlink("$excl_file") == 1 or die("Cannot delete '$excl_file': $!")
    }
    if (-e "$kill_file") {
        unlink("$kill_file") == 1 or die("Cannot delete '$kill_file': $!")
    }
    if (-e "$summary_file") {
        unlink("$summary_file") == 1 or die("Cannot delete '$summary_file': $!")
    }
    if (-e "$test_map_file") {
        unlink("$test_map_file") == 1 or die("Cannot delete '$test_map_file': $!")
    }
}



# Función para copiar archivos y directorios
sub copy_directory {
    my $origen = shift;
    my $destino = shift;

    find(sub {
        my $ruta_relativa = $File::Find::name;
        $ruta_relativa =~ s/^\Q$origen\E//;

        my $destino_completo = $destino . $ruta_relativa;

        if (-d $_) {
            # Si es un directorio, crea el directorio en el destino
            unless (-d $destino_completo) {
                make_path($destino_completo) or die "No se pudo crear el directorio '$destino_completo': $!\n";
            }
        } else {
            # Si es un archivo, copia el archivo al destino
            copy($_, $destino_completo) or die "No se pudo copiar '$_' a '$destino_completo': $!\n";
        }
    }, $origen);
}


sub copy_directory_and_only_java_files {
    my $origen = shift;
    my $destino = shift;

    find(sub {
        my $ruta_relativa = $File::Find::name;
        $ruta_relativa =~ s/^\Q$origen\E//;

        my $destino_completo = $destino . $ruta_relativa;

        if (-d $_) {
            # Si es un directorio, crea el directorio en el destino si no existe
            unless (-d $destino_completo) {
                make_path($destino_completo) or die "No se pudo crear el directorio '$destino_completo': $!\n";
            }
        } else {
            # Si es un archivo .java, copia el archivo al destino
            if (/\.java$/) {
                copy($_, $destino_completo) or die "No se pudo copiar el archivo java '$_' a '$destino_completo': $!\n";
            }
        }
    }, $origen);
}


sub find_file_java_test {
    my $test_dir = shift;
    my $focal_java_class = shift;

    my $file_java_test_to_return;
    # Buscamos la clase de test dentro del archivo comprimido
    find(sub {
        # if (-f && /\.java$/) {
        return unless -f; # Retorna si no es un file
        return unless /\.java$/; # Retorna si no es un archivo.java
        if (!/scaffolding\.java$/) { # Obtenemos los que no terminan en scaffolding.java
            my $java_file_find = $File::Find::name;
            my $test_java_file_absolute_path = $java_file_find;
            # Eliminamos la parte de la ruta base para obtener la ruta relativa
            $java_file_find =~ s/^$test_dir\/?//;     # Only com/google/gson/MyClass_ESTests.java
            $java_file_find =~ s/\//./g;              # Reemplaza '/' por '.'
            # $java_file_find =~ s/\/\././g;
            # $test_java_class = $java_file_find;
            # $test_java_class =~ s/\.java$//; 
            if ($USE_SCAFFOLDING) {
                $java_file_find =~ s/_ESTest\.java$//;  # Elimina '_ESTest.java' al final de la cadena
            } else {
                $java_file_find =~ s/Test\.java$//;     # Elimina 'Test.java' al final de la cadena
            }

            my $focal_java_class_find = $java_file_find;
            if ($focal_java_class eq $focal_java_class_find) {
                $file_java_test_to_return = $test_java_file_absolute_path;
            }
        }
    }, $test_dir);

    return $file_java_test_to_return;
}


sub _update_test_class_name {
    my ($actual_class_name, $new_class_name, $actual_java_file, $new_java_file) = @_;

    open(IN, "<$actual_java_file") or die $!; my @lines_class = <IN>; close IN;

    if (! -e "$actual_java_file.bak") {
        copy("$actual_java_file", "$actual_java_file.bak") or die "Cannot backup file ($actual_java_file): $!";
    }

    my $actual_class_name_escaped = quotemeta($actual_class_name);

    # Reemplaza la línea con el método generado
    for (@lines_class) {
        # next unless /public\s*void\s*(test.*)\s*\(\s*\).*/;
        next unless /public class $actual_class_name_escaped \{\n/;
        s{public class $actual_class_name_escaped \{\n}{public class $new_class_name \{\n} if /public class $actual_class_name_escaped \{\n/;
        last;
    }

    # Escribe en el archivo
    open my $fh, '>', $new_java_file or die "No se pudo abrir el archivo $new_java_file para actualizar el nombre de clase $new_class_name: $!";
    print $fh @lines_class;
    close $fh;

    unlink $actual_java_file or die "No se pudo eliminar el archivo $actual_java_file: $!\n";
}


sub _mutation_analysis_ext {
    @_ >= 4 or die $ARG_ERROR;
    my ($proj, $dir, $include, $log_file, $exclude_file, $single_test) = @_;
    my $log = "-logfile $log_file";
    my $exclude = defined $exclude_file ? "-Dmajor.exclude=$exclude_file" : "";

    my $basedir = $proj->{prog_root};

    my $single_test_opt = "";
    if (defined $single_test) {
        $single_test =~ /([^:]+)::([^:]+)/ or die "Wrong format for single test!";
        $single_test_opt = "-Dtest.entry.class=$1 -Dtest.entry.method=$2";
    }

    return $proj->_call_major("mutation.gen.test",
                            "-Dd4j.test.dir=$dir -Dd4j.test.include=$include " .
                            "-Dmajor.kill.log=$basedir/$KILL_FILE " .
                            "$log $exclude $single_test_opt");
}
