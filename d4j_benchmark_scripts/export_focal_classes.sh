#!/usr/bin/env bash

repos=(Csv Cli Lang Chart Gson)
#repos=(Gson)

repo_type=f

export_focal_classes() {
	json_array=[]
	for repo in ${repos[@]}; do
		results=`defects4j query -p ${repo} -q "classes.modified"` # This prints list (of records) of classes modified (fixed) by bug.id by each repo and save in @results
		results=($(echo "$results" | sed 's/\"//g'))
		for res in ${results[@]}; do
			bugid=$(echo $res | cut -d "," -f 1)
			focal_classes=$(echo $res | cut -d "," -f 2)
			# IFS=';' read -ra focal_classes <<< "$focal_classes"
			focal_classes=$(echo $focal_classes | jq -R -s -c 'split(";")')
			revision=${repo}_${bugid}_${repo_type}
			echo "project: $revision class: ${focal_classes[@]}"
			json_array=$(echo "$json_array" | jq --arg project "${revision}" --argjson classes "$focal_classes" '. += [{"project": $project, "classes": $classes}]')
		done
	done
	echo $json_array > focal_classes.json
}

export_focal_classes
