#!/usr/bin/env bash

repos=(Csv Cli Lang Chart Gson)
#repos=(Gson)

#revisions_dir=E:/000_Tesis/defects4j/framework/custom/defects4j_revisions
revisions_dir=/tmp/defects4j_revisions
repo_type=f

export_revisions() {
	for repo in ${repos[@]}; do
		cd $revisions_dir
		results=(`defects4j query -p ${repo} -q "bug.id"`) # This prints list (of numbers) of bug versions by each repo and save in @results
		for id in ${results[@]}; do
			repo_name=${repo}_${id}_${repo_type}
			repo_path=${revisions_dir}/${repo_name}
			( defects4j checkout -p $repo -v ${id}${repo_type} -w ${revisions_dir}/${repo_name} ) || continue
			cd $repo_name && defects4j compile
		done
	done
}


export_revisions
