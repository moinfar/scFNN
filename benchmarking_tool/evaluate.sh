#!/bin/bash

if [ "$1" = "all" ] ||  [ "$1" = "cell-cycle" ];
then
	for method in $(ls ~/code/test\ results/ | awk '!/.*.sh/')
	do
		filename=~/code/test\ results/${method}/cell_cycle_hq.csv.gz;
		result=~/code/test\ results/${method}/cell_cycle_hq.csv.result.d;
		echo "    >>> Checking $filename ...";
		if [ -e "$filename" ];
		then
			if [ ! -e "$result/result.txt" ] ||  [ "$2" = "-f" ];
			then
			        echo "python run.py cell_cycle_hq -D -S 123 evaluate -i "$filename" -r "$result" cell-cycle";
		        	python run.py cell_cycle_hq -D -S 123 evaluate -i "$filename" -r "$result" cell-cycle ;
			else
		                echo "    >>> Result is already evaluated"
			fi
		else
			echo ">>> No Result found !!!"
		fi
	done
fi



if [ "$1" = "all" ] ||  [ "$1" = "clustering" ];
then
	for ds in cortex baron_human baron_mouse pollen_hq pollen_lq
	do
		printf "\n\n\n--------------------------------------------\n\n\n"

		for method in $(ls ~/code/test\ results/ | awk '!/.*.sh/')
		do
		        filename=~/code/test\ results/${method}/clustering_${ds}.csv.gz;
		        result=~/code/test\ results/${method}/clustering_${ds}.csv.result.d;
		        echo "    >>> Checking $filename ...";
		        if [ -e "$filename" ];
		        then
		                if [ ! -e "$result/result.txt" ] ||  [ "$2" = "-f" ];
		                then
		                        echo "python run.py clustering_${ds} -D -S 123 evaluate -i "$filename" -r "$result" clustering";
		                        python run.py clustering_${ds} -D -S 123 evaluate -i "$filename" -r "$result" clustering ;
		                else
		                        echo "    >>> Result is already evaluated"
		                fi
		        else
		                echo ">>> No Result found !!!"
		        fi
		done
	done
fi




if [ "$1" = "all" ] ||  [ "$1" = "random-mask" ];
then
	for ds in 10xPBMC4k_1 10xPBMC4k_2 cortex_1 cortex_2 cell_cycle_1 cell_cycle_2 baron_human_1 baron_human_2 baron_mouse_1 baron_mouse_2 pollen_hq_1 pollen_hq_2 pollen_lq_1 pollen_lq_2 cite_cd8_1 cite_cd8_2
	do
		printf "\n\n\n--------------------------------------------\n\n\n"

		for method in $(ls ~/code/test\ results/ | awk '!/.*.sh/')
		do
		        filename=~/code/test\ results/${method}/random_mask_${ds}.csv.gz;
		        result=~/code/test\ results/${method}/random_mask_${ds}.csv.result.d;
		        echo "    >>> Checking $filename ...";
		        if [ -e "$filename" ];
		        then
		                if [ ! -e "$result/result.txt" ] ||  [ "$2" = "-f" ];
		                then
		                        echo "python run.py random_mask_${ds} -D -S 123 evaluate -i "$filename" -r "$result" random-mask";
		                        python run.py random_mask_${ds} -D -S 123 evaluate -i "$filename" -r "$result" random-mask ;
		                else
		                        echo "    >>> Result is already evaluated"
		                fi
		        else
		                echo ">>> No Result found !!!"
		        fi
		done
	done
fi




if [ "$1" = "all" ] ||  [ "$1" = "down-sample" ];
then
	for ds in 10xPBMC4k_1 10xPBMC4k_2 cortex_1 cortex_2 cell_cycle_1 cell_cycle_2 baron_human_1 baron_human_2 baron_mouse_1 baron_mouse_2 pollen_hq_1 pollen_hq_2 pollen_lq_1 pollen_lq_2 cite_cd8_1 cite_cd8_2
	do
		printf "\n\n\n--------------------------------------------\n\n\n"

		for method in $(ls ~/code/test\ results/ | awk '!/.*.sh/')
		do
			filename=~/code/test\ results/${method}/down_sample_${ds}.csv.gz;
			result=~/code/test\ results/${method}/down_sample_${ds}.csv.result.d;
			echo "    >>> Checking $filename ...";
			if [ -e "$filename" ];
			then
			        if [ ! -e "$result/result.txt" ] ||  [ "$2" = "-f" ];
			        then
			                echo "python run.py down_sample_${ds} -D -S 123 evaluate -i "$filename" -r "$result" down-sample";
			                python run.py down_sample_${ds} -D -S 123 evaluate -i "$filename" -r "$result" down-sample ;
			        else
			                echo "    >>> Result is already evaluated"
			        fi
			else
			        echo ">>> No Result found !!!"
			fi
		done
	done
fi




if [ "$1" = "all" ] ||  [ "$1" = "paired-data" ];
then
	printf "\n\n\n--------------------------------------------\n\n\n"
	for method in $(ls ~/code/test\ results/ | awk '!/.*.sh/')
	do
		filename=~/code/test\ results/${method}/paired_data_pollen.csv.gz;
		result=~/code/test\ results/${method}/paired_data_pollen.csv.result.d;
		echo "    >>> Checking $filename ...";
		if [ -e "$filename" ];
		then
		        if [ ! -e "$result/result.txt" ] ||  [ "$2" = "-f" ];
		        then
		                echo "python run.py paired_data_pollen -D -S 123 evaluate -i "$filename" -r "$result" paired-data";
		                python run.py paired_data_pollen -D -S 123 evaluate -i "$filename" -r "$result" paired-data ;
		        else
		                echo "    >>> Result is already evaluated"
		        fi
		else
		        echo ">>> No Result found !!!"
		fi
	done
fi





if [ "$1" = "all" ] ||  [ "$1" = "cite-seq" ];
then
	for ds in cd8 pbmc cbmc
	do
		printf "\n\n\n--------------------------------------------\n\n\n"

		for method in $(ls ~/code/test\ results/ | awk '!/.*.sh/')
		do
			filename=~/code/test\ results/${method}/cite_seq_${ds}.csv.gz;
			result=~/code/test\ results/${method}/cite_seq_${ds}.csv.result.d;
			echo "    >>> Checking $filename ...";
			if [ -e "$filename" ];
			then
				if [ ! -e "$result/result.txt" ] ||  [ "$2" = "-f" ];
				then
				        echo "python run.py cite_seq_${ds} -D -S 123 evaluate -i "$filename" -r "$result" cite-seq";
				        python run.py cite_seq_${ds} -D -S 123 evaluate -i "$filename" -r "$result" cite-seq ;
				else
				        echo "    >>> Result is already evaluated"
				fi
			else
				echo ">>> No Result found !!!"
			fi
		done
	done
fi


