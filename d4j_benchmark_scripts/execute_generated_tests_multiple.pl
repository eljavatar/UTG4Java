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
use Data::Dumper;
# Install module Text::CSV
# cpan Text::CSV
use IPC::Open3; # Para ejecutar un proceso y capturar su salida
use Symbol 'gensym';

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
getopts('p:d:o:v:s:t:f:AD', \%cmd_opts) or pod2usage(1);

pod2usage(1) unless defined $cmd_opts{p} and 
                    defined $cmd_opts{d} and 
                    defined $cmd_opts{o};


my $SUITE_DIR = abs_path($cmd_opts{d}); # /generated-tests/Gson
my $OUT_DIR = abs_path($cmd_opts{o});
my $PID = $cmd_opts{p};
my $VID = $cmd_opts{v} if defined $cmd_opts{v};
my $TEST_SRC = $cmd_opts{s} if defined $cmd_opts{s};
my $INCL = $cmd_opts{f} // "*.java";
my $RM_ASSERTS = defined $cmd_opts{A} ? 1 : 0;
# Enable debugging if flag is set
$DEBUG = 1 if defined $cmd_opts{D};

# Guarda el tiempo de inicio
my $start_time = [gettimeofday];

# Check format of target version id
if (defined $VID) {
    Utils::check_vid($VID);
}

my @list;
opendir(DIR, $SUITE_DIR) or die "Could not open directory: $SUITE_DIR!";
my @entries = readdir(DIR);
closedir(DIR);

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

    $src = "generated_test_$sid";

    my $vid_num = $vid;
    $vid_num =~ s/[^0-9]//g; # Reemplaza la parte no nuérica por una cadena vacía
    #$vid_num = int($vid_num);
    #print "\n\n vid: $vid --- vid_num: $vid_num \n\n";

    push(@list, {name => $_, pid => $pid, vid=>$vid, vid_num=>$vid_num, src=>$src, tid=>$tid, sid=>$sid});
}
# Ordenamos la lista
#@list = sort { $a->{name} cmp $b->{name} } @list;
@list = sort {
    # Comparar por 'version' primero
    #$a->{vid} cmp $b->{vid}
    $a->{vid_num} <=> $b->{vid_num}
    ||
    # Comparar por 'number_method_by_version' si 'version' es igual
    #$a->{tid} <=> $b->{tid}
    $a->{tid} <=> $b->{tid}
    ||
    $a->{sid} <=> $b->{sid}
} @list;


# Set up project
my $TMP_DIR = Utils::get_tmp_dir($cmd_opts{t}); # This is $WORK_DIR
system("mkdir -p $TMP_DIR");

#my $PATH_LOGS = "/tmp";
my $PATH_LOGS = "$OUT_DIR/results";
#my $PATH_COVERAGE = "$OUT_DIR/coverage";
#my $PATH_INSTRUMENTS = "$PATH_COVERAGE/instruments";
my $PATH_INSTRUMENTS = "/tmp/instruments";
#if (!-e "$PATH_LOGS/results") {
#    system("mkdir -p $PATH_LOGS/results");
#}
if (!-e "$PATH_LOGS") {
    system("mkdir -p $PATH_LOGS");
}
#if (!-e "$PATH_COVERAGE") {
#    system("mkdir -p $PATH_COVERAGE");
#}
if (!-e "$PATH_INSTRUMENTS") {
    system("mkdir -p $PATH_INSTRUMENTS");
}

my $COMPILE_LOG = Log::create_log("$PATH_LOGS/${PID}_tests_execution.compile.log");
my $RUN_LOG     = Log::create_log("$PATH_LOGS/${PID}_tests_execution.run.log");
my $SUMMARY_LOG = Log::create_log("$PATH_LOGS/${PID}_tests_execution.summary.log");
my $RESULTS_LOG = Log::create_log("$PATH_LOGS/${PID}_tests_execution.results.log");


# Line separator
my $sep = "-"x80;

# Log current time
$SUMMARY_LOG->log_time("Start tests execution");
$SUMMARY_LOG->log_msg("- Found " . scalar(@list) . " test archive(s)");
$SUMMARY_LOG->log_msg("\n");


#Syntax error: Validate with AST (Abstract Syntax Tree)
#Build error: Validate of compile
#Failing: Compilan pero fallan en la ejecución (fail for incorrect assertions, or wrong expected beavior, e.. the test expects an exception which is not raised)
#Passing: Syntactically correct, compilable and execute without failing
#Correct: Subconjunto de los tests Passing que cubren el método focal correcto proporcionado como entrada: Has coverage


my @list_results = ();
my @list_coverage = ();

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
# my $num_with_branch_covergage = 0;


# Agrupamos por project, version, test_id
my %groups;

# Itera sobre cada objeto en la lista de objetos
for my $object (@list) {
    # Crea una clave de grupo con los valores de 'project', 'version' y 'test_id'
    my $group_key = join '|', @{$object}{qw(pid vid vid_num tid)};

    # Agrega el objeto al grupo correspondiente
    push @{$groups{$group_key}}, $object;
}


# Itera sobre cada grupo en el hash de grupos
while (my ($group_key, $list_tests_by_group) = each %groups) {
    # Divide la clave del grupo en 'project', 'version' y 'test_id'
    my ($g_project, $g_version, $g_version_num, $g_test_id) = split(/\|/, $group_key);

    #Build object res
    my $res = {
        name => "${g_project}_${g_version}_${g_test_id}",
        project => $g_project,
        version => $g_version, 
        version_num => $g_version_num, 
        test_id => $g_test_id,
        correct_syntax => "No",
        compilable => "No",
        executable => "No",
        passing_test => "No",
        correct_test => "No"
        # branch_covered => "No"
        #modified_class => $class,
        #test_class => $test_class_scaffolding_name,
        #method_to_test => $method_to_test
    };

    my $syntax = 0;
    my $compile = 0;
    my $execute = 0;
    my $passing = 0;
    my $fail_by_class = 0;
    my $fail_by_method = 0;
    my $fail_by_class_and_method = 0;

    suite: foreach (@$list_tests_by_group) {
        # Obtenemos la info de cada archivo de tests
        my $name = $_->{name};
        my $pid  = $_->{pid};
        my $vid  = $_->{vid};
        my $src  = $_->{src};
        my $tid  = $_->{tid};
        my $sid  = $_->{sid};

        # Obtenemos el proyecto
        my $project = Project::create_project($pid);
        $project->{prog_root} = $TMP_DIR;

        printf ("$sep\n$name\n$sep\n");

        # Validamos la versión del proyecto
        $project->checkout_vid($vid);

        my $project_test_dir = "$TMP_DIR/$src";
        # Extract generated tests into temp directory
        Utils::extract_test_suite("$SUITE_DIR/$name", $project_test_dir)
            or die "Cannot extract test suite!";
        
        my @instruments_to_coverage;
        my $test_java_file = "";
        my $test_java_class = "";
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
                $java_file_find =~ s/_ESTest\.java$//;   # Elimina '_ESTest.java' al final de la cadena
                #push @list_to_coverage_global, $java_file_find;
                push @instruments_to_coverage, $java_file_find;
            }
            #push @list_to_coverage, $File::Find::name;
        }, $project_test_dir);

        my $file_instrument_classes = "${PATH_INSTRUMENTS}/${pid}_${vid}_instruments_${tid}_${sid}.txt";
        open my $fh, '>', $file_instrument_classes or die "No se pudo abrir el archivo $file_instrument_classes: $!";
        print $fh @instruments_to_coverage;
        close $fh;

        printf ("\n Clase a analizar: $test_java_file\n\n");
        my $has_syntax_error = _get_result_parser_utils($test_java_file, "HAS_ERROR"); # Error | No Error
        my $test_method_name = _get_result_parser_utils($test_java_file, "GET_METHOD_NAME");

        my $test_canonical_name = "${test_java_class}::$test_method_name";

        my $tag_version_tid_sid = "v_${vid} -> ${tid}--${sid}";

        print "Resultado has_syntax_error ($tag_version_tid_sid): $has_syntax_error\n";
        print "Resultado test_method_name ($tag_version_tid_sid): $test_method_name\n";
        print "Resultado test_canonical_name ($tag_version_tid_sid): $test_canonical_name\n\n";


        # Validate Syntax error
        if ($has_syntax_error eq "Error") {
            $SUMMARY_LOG->log_msg(" - Syntax error in test method(s): $name");
            $SUMMARY_LOG->log_msg("   --- $test_canonical_name");
            $SUMMARY_LOG->log_msg("");

            #$num_syntax_error += 1;

            #push(@list_results, $res);
            print "\n\n";

            next suite;
        }
        if (!$syntax) {
            $num_syntax_correct += 1;
            $res->{correct_syntax} = "Yes";
            $syntax = 1;
        }
        print "CORRECT SYNTAX for test ($tag_version_tid_sid): $test_canonical_name\n\n";


        # Temporary log file to monitor uncompilable tests
        my $compilation_log = Log::create_log("$TMP_DIR/comp_tests.log", ">")->{file_name};

        # Check for compilation errors
        if (! $project->compile_ext_tests($project_test_dir, $compilation_log)) {
            #$COMPILE_LOG->log_file("=> Compilation issues: $name", $compilation_log);
            $COMPILE_LOG->log_msg("=> Compilation issues: $name\n");
            open(my $log_fh, '<', $compilation_log) or die "No se pudo abrir el archivo $compilation_log: $!";
            while (my $line = <$log_fh>) {
                # Omitimos líneas que comienzan con [javac] warning:
                # y las que comienzan con [javac] It is recommended that the compiler be upgraded
                if ($line =~ /javac.*warning:/ || $line =~ /javac.*It is recommended that the compiler be upgraded/) {
                    next;
                }
                chomp($line); # Usamos chomp para eliminar el salto de línea al final (\n)
                $COMPILE_LOG->log_msg($line);
            }
            close($log_fh);
            $COMPILE_LOG->log_msg("\n\n\n\n\n");

            my ($n_uncompilable_tests, $n_uncompilable_test_classes) = _rm_classes($compilation_log, $src, $name, $test_method_name);

            #$num_compile_error += 1;

            #push(@list_results, $res);
            print "\n\n";

            next suite;
        }
        if (!$compile) {
            $num_compile_correct += 1;
            $res->{compilable} = "Yes";
            $compile = 1;
        }
        print "COMPILABLE for test ($tag_version_tid_sid): $test_canonical_name\n\n";


        # Temporary log file to monitor failing tests
        my $log_run_tests = Log::create_log("$TMP_DIR/run_tests.log", ">")->{file_name};

        # Check for errors of runtime system
        if (! $project->run_ext_tests($project_test_dir, "$INCL", $log_run_tests)) {
            $SUMMARY_LOG->log_file(" - Test not executable: $name", $log_run_tests);

            #$num_execute_error += 1;

            #push(@list_results, $res);
            print "\n\n";

            next suite;
        }
        if (!$execute) {
            $num_execute_correct += 1;
            $res->{executable} = "Yes";
            $execute = 1;
        }
        print "EXECUTABLE for test ($tag_version_tid_sid): $test_canonical_name\n\n";


        # Check failing test classes and methods
        my $list_failings = Utils::get_failing_tests($log_run_tests) or die;
        if (scalar(@{$list_failings->{classes}}) != 0 || scalar(@{$list_failings->{methods}}) != 0) {
            my $has_failing_class = 0;

            if (scalar(@{$list_failings->{classes}}) != 0) {
                $has_failing_class = 1;
                $RUN_LOG->log_file("Failing test class: $name", $log_run_tests);
                $RUN_LOG->log_msg("");

                $SUMMARY_LOG->log_msg(" - Failing test class: $name");
                $SUMMARY_LOG->log_msg(join("\n", @{$list_failings->{classes}}));
                $SUMMARY_LOG->log_file(" - Stack traces:", $log_run_tests);
                #$SUMMARY_LOG->log_msg("");

                #$num_failing_by_class += 1;
                $fail_by_class = 1;
                print "FAILING TEST (by class) for test ($tag_version_tid_sid): $test_canonical_name\n";
            }

            if (scalar(@{$list_failings->{methods}}) != 0) {
                $RUN_LOG->log_file(scalar(@{$list_failings->{methods}}) . " broken test method(s): $name", $log_run_tests);
                $RUN_LOG->log_msg("");

                my $msg_prefix = " - ";
                if ($has_failing_class) {
                    #$num_failing_by_class_and_method += 1;
                    $fail_by_class_and_method = 1;
                    $msg_prefix = " --> And ";
                }
                $SUMMARY_LOG->log_msg("${msg_prefix}Finding " . scalar(@{$list_failings->{methods}}) . " broken test method(s): $name");
                $SUMMARY_LOG->log_msg(join("\n", @{$list_failings->{methods}}));
                #$SUMMARY_LOG->log_msg("");
                
                #$num_failing_by_method += 1;
                $fail_by_method = 1;
                print "FAILING TEST (by method) for test ($tag_version_tid_sid): $test_canonical_name\n";

                # push(@list_results, $res);
            }

            print "\n\n";
            $RUN_LOG->log_msg("\n");
            $SUMMARY_LOG->log_msg("");

            #$num_failing += 1;
            #push(@list_results, $res);

            next suite;
        }
        if (!$passing) {
            $num_passing += 1;
            $res->{passing_test} = "Yes";
            $passing = 1;
        }
        print "PASSING ASSERTS for test ($tag_version_tid_sid): $test_canonical_name\n\n";
        # push(@list_results, $res);


        $SUMMARY_LOG->log_msg(" - PASSED CORRECTLY test method(s): $name");
        $SUMMARY_LOG->log_msg("$test_canonical_name");
        $SUMMARY_LOG->log_msg("");


        #El coverage lo ejecutamos solo sobre los tests que pasan para validar si cubren el método
        my $src_dir = $project->src_dir($vid);
        my $SINGLE_TEST = $test_canonical_name;
        # Clean temporary files that hold test results
        my $fail_tests_log = "$TMP_DIR/$FILE_FAILING_TESTS";
        Utils::clean_test_results($TMP_DIR);

        # El uso de $SINGLE_TEST falla en casos como por ejemplo $Gson$Types
        #my $cov_results = Coverage::coverage_ext($project, $file_instrument_classes, $src_dir, $project_test_dir, "*.java", $fail_tests_log, $SINGLE_TEST);
        my $cov_results = Coverage::coverage_ext($project, $file_instrument_classes, $src_dir, $project_test_dir, "*.java", $fail_tests_log);
        #Utils::clean_test_results($TMP_DIR);

        $cov_results->{line_coverage} = $cov_results->{lines_covered}/$cov_results->{lines_total}*100;
        $cov_results->{condition_coverage} = ($cov_results->{branches_total} == 0 ? 0 : $cov_results->{branches_covered}/$cov_results->{branches_total}*100);

        $cov_results->{name} = $name;
        $cov_results->{project} = $pid;
        $cov_results->{version} = $vid;
        $cov_results->{test_id} = $tid;

        push(@list_coverage, $cov_results);

        if ($cov_results->{lines_covered} > 0) {
            $num_correct += 1;
            $res->{correct_test} = "Yes";

            print "CORRECT TEST (LINE COVERED) ($tag_version_tid_sid): $test_canonical_name\n";
        }
        # if ($cov_results->{branches_covered} > 0) {
        #     $num_with_branch_covergage += 1;
        #     $res->{branch_covered} = "Yes";

        #     print "CORRECT TEST (BRANCH_COVERED): $test_canonical_name\n";
        # }
        
        print "\n\n";

        unlink $file_instrument_classes or die "No se pudo eliminar el archivo $file_instrument_classes: $!\n";

        last;
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
}



# Log current time
$SUMMARY_LOG->log_msg("");
$SUMMARY_LOG->log_time("End tests execution");
$SUMMARY_LOG->close();
$COMPILE_LOG->close();
$RUN_LOG->close();


# Ordenamos la lista de resultados
@list_results = sort {
    # Comparar por 'version' primero
    #$a->{vid} cmp $b->{vid}
    $a->{version_num} <=> $b->{version_num}
    ||
    # Comparar por 'number_method_by_version' si 'version' es igual
    #$a->{tid} <=> $b->{tid}
    $a->{test_id} <=> $b->{test_id}
} @list_results;


# Creamos una instancia de Text::CSV
my $csv = Text::CSV->new({ binary => 1, eol => "\n" });

# Abrimos un archivo para escribir
#open my $file, ">", "$PATH_LOGS/results/${PID}_report_execution.csv" or die "Cannot write methods hashes in CSV";
open my $file, ">", "$PATH_LOGS/${PID}_report_execution.csv" or die "Cannot write methods hashes in CSV";
# Especificamos el orden de las columnas
my @sort_columns = (
    'name', 
    'project', 
    'version', 
    'version_num',
    'test_id', 
    'correct_syntax',
    'compilable', 
    'executable', 
    'passing_test',
    'correct_test'
);

# Escribimos los nombres de las columnas en la primera línea
$csv->print($file, \@sort_columns);
foreach my $hash (@list_results) {
    # Escribimos cada valor en el archivo
    $csv->print($file, [@{$hash}{@sort_columns}]);
}
# Cerramos el archivo
close $file;


open(CSV, ">$PATH_LOGS/${PID}_summary_coverage.csv") or die "Cannot write output csv file $!";
print(CSV "Name,Project,Version,TestId,LinesTotal,LinesCovered,RateLineCoverage,ConditionsTotal,ConditionsCovered,RateConditionCoverage\n");
foreach my $cov (@list_coverage) {
    printf(CSV "%s,%s,%s,%d,%d,%d,%s,%d,%d,%s\n",
            $cov->{name},
            $cov->{project},
            $cov->{version},
            $cov->{test_id},
            $cov->{lines_total},
            $cov->{lines_covered},
            #$cov->{line_coverage}, -> %f
            sprintf("%.2f", $cov->{line_coverage}),
            $cov->{branches_total},
            $cov->{branches_covered},
            #$cov->{condition_coverage} -> %f
            sprintf("%.2f", $cov->{condition_coverage})
    );
}
close(CSV);


#my $total_tests = scalar(@list);
my $total_tests = scalar(%groups);


print(STDERR "\n\n");
print(STDERR "SUMMARY RESULTS PROJECT: $PID\n\n");
print(STDERR "Total tests: $total_tests\n\n");

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
print(STDERR "\n\n");


$RESULTS_LOG->log_msg("- Total tests: $total_tests");
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

$RESULTS_LOG->close();


# Imprimimos el summary de los resultados
my $file_all_summary_results = "$PATH_LOGS/all_summary_results.csv";
my $already_exists_file_summary = 0;
if (-e "$file_all_summary_results") {
    $already_exists_file_summary = 1;
}

# Con '>' se sobreescribe el archivo, con '>>' se agregan registros al archivo existente
open(CSV, ">>$file_all_summary_results") or die "Cannot write output $file_all_summary_results csv file $!";
if (! $already_exists_file_summary) {
    # Imprimimos el head solo si el archivo no existía antes
    print(CSV "project,total_tests,syntax_error,rate_syntax_error,compile_error,rate_compile_error,execute_error,rate_execute_error,failing_tests,rate_failing_tests," . 
                "syntax_correct,rate_syntax_correct,compile_correct,rate_compile_correct,passing_tests,rate_passing_tests,correct_tests,rate_correct_tests\n");
}
printf(CSV "%s,%d,%d,%s,%d,%s,%d,%s,%d,%s,%d,%s,%d,%s,%d,%s,%d,%s\n",
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
        sprintf("%.2f", $rate_correct)
);
close(CSV);



# Clean up
system("rm -rf $PATH_INSTRUMENTS");
system("rm -rf $TMP_DIR") unless $DEBUG;

# Calcula el tiempo transcurrido
my $elapsed = tv_interval($start_time);
# Convierte el tiempo a horas, minutos y segundos
my $hours   = int($elapsed / 3600);
my $minutes = int(($elapsed % 3600) / 60);
my $seconds = int($elapsed % 60);
printf("\nTiempo de ejecucion project ${PID}: %02d:%02d:%02d\n\n", $hours, $minutes, $seconds);



#
# Remove uncompilable test cases based on the compiler's log (if there
# is any issue non-related to any test case, the correspondent source
# file is removed)
#
sub _rm_classes {
    my ($comp_log, $src, $name, $test_method_name) = @_;

    open(LOG, "<$comp_log") or die "Cannot read compiler log!";
    $SUMMARY_LOG->log_msg(" - Catching uncompilable test method(s): $name");
    my $num_uncompilable_test_classes = 0;
    my @uncompilable_tests = ();
    my $error;

    while (<LOG>) {
        #my $removed = 0;
        my $find_test_error = 0;

        # Find file names in javac's log: [javac] "path"/"file_name".java:"line_number": error: "error_text"
        #next unless /javac.*($TMP_DIR\/$src\/(.*\.java)):(\d+):.*error/;
        #next unless /javac.*($TMP_DIR\/$src\/(.*\.java)):(\d+):.*(error:.([^-]+))/;
        next unless /javac.*($TMP_DIR\/$src\/(.*\.java)):(\d+):.*(error:.([^\.]+))/;
        my $file = $1;
        my $class = $2;
        my $line_number = $3;
        $error = $5;

        #print(STDERR "File before: $file\n");
        #print(STDERR "Class inmediatly: $class\n");

        # /javac.*(/tmp/checkouts/replace_generated_tests.pl_50685_1706071672\/generated-tests\/(.*\.java)):(\d+):.*error/

        # Skip already removed files
        next unless -e $file;

        $class =~ s/\.java$//;
        #print(STDERR "Class after .java: $class\n");

        $class =~ s/\//\./g;
        #print(STDERR "Class after puntos: $class\n");

        my $test_name = "";
        if ($test_method_name eq "There Are No Methods" || $test_method_name eq "More Than One") {
            # To which test method does the uncompilable line belong?
            open(JAVA_FILE, $file) or die "Cannot open '$file' file!";
            my $line_index = 0;
            while (<JAVA_FILE>) {
                ++$line_index;
                next unless /public\s*void\s*(test.*)\s*\(\s*\).*/;
                my $t_name = $1;

                if ($line_index > $line_number) {
                    last;
                }

                $test_name = $t_name;
                #$removed = 1;
                $find_test_error = 1;
            }
            close(JAVA_FILE);
        } else {
            $test_name = $test_method_name;
            #$removed = 1;
            $find_test_error = 1;
        }

        #print(STDERR "File after: $file\n");

        #if (! $removed) {
        if (! $find_test_error) {
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
            my $test_canonical_name = "     --- ${class}::${test_name}";
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
        print "Errores del script Python: $error_output\n";
    }

    return $res_script_python;
}



sub _run_coverage() {

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

    # $cov_results->{name} = $name;
    # $cov_results->{project} = $pid;
    # $cov_results->{version} = $vid;
    # $cov_results->{test_id} = $tid;

    # push(@list_coverage, $cov_results);

    #unlink $file_instrument_classes or die "No se pudo eliminar el archivo $file_instrument_classes: $!\n";

    # printf("%18s: %d\n",     "Lines total",        $cov_results->{lines_total});
    # printf("%18s: %d\n",     "Lines covered",      $cov_results->{lines_covered});
    # printf("%18s: %d\n",     "Conditions total",   $cov_results->{branches_total});
    # printf("%18s: %d\n",     "Conditions covered", $cov_results->{branches_covered});
    # printf("%18s: %.1f%%\n", "Line coverage",      $cov_results->{lines_covered}/$cov_results->{lines_total}*100);
    # printf("%18s: %.1f%%\n", "Condition coverage", ($cov_results->{branches_total} == 0 ? 0 : $cov_results->{branches_covered}/$cov_results->{branches_total}*100));

}